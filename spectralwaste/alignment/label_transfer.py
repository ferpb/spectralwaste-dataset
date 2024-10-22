try:
    import COTR
    from COTR import models, utils
    from COTR.inference.sparse_engine import SparseEngine
except Exception as e:
    raise ImportError('Install COTR to use the alignment module: pip install -e third_party/COTR')

import numpy as np
import easydict
import json
import torch
import cv2
import os
import functools

class LabelTransferModel:
    def __init__(self, model_path=None, verbose=False):
        self.verbose = verbose

        if not model_path:
            # Default model path
            model_path = os.path.join(os.path.dirname(COTR.__file__), '../out/default')

        params = json.load(open(os.path.join(model_path, 'params.json')))
        params = easydict.EasyDict(params)
        weights = torch.load(os.path.join(model_path, 'checkpoint.pth.tar'))['model_state_dict']

        model = models.build_model(params)
        utils.utils.safe_load_weights(model, weights)
        model.cuda()
        model.eval()
        self.engine = SparseEngine(model, 32, mode='tile', verbose=verbose)

    def transfer_instance_labels(self, img1, img2, instance_labels):
        # Transfer rgb masks
        transferred_masks = []
        for mask in instance_labels['masks']:
            transferred_mask = self.transfer_mask(img1, img2, mask)
            transferred_masks.append(transferred_mask)
        return dict(masks=np.stack(transferred_masks), categories=instance_labels['categories'])

    def transfer_mask(self, img1, img2, mask1, distances_thres=[10, 20, 30, 40, 50], return_extra=False):
        """
        Transfer labels for img1 to the corresponding img2
        img1 y img2: np.array[np.uint8]
        mask1: binary mask
        """
        assert img1.shape[:2] == img2.shape[:2]
        final_mask = np.zeros_like(mask1, dtype=np.uint8)

        # Get connected components of the mask
        components_masks = self._get_connected_components(mask1)

        if self.verbose:
            print(f'Found {len(components_masks)} components')

        extra = []

        # Transfer each connected component
        for component_mask in components_masks:
            max_det = 0
            best_H = None
            best_points = None
            best_corrs = None
            best_points_extra = None

            # Find best transformation sampling different number of points
            for distance_thres in distances_thres:
                if self.verbose:
                    print(f'Distance threshold: {distances_thres}')

                # Find contour points for the component
                points, points_extra = self._get_descriptive_points(component_mask, distance_thres=distance_thres, return_extra=True)

                if points.shape[0] == 0:
                    continue

                # Find correspondences using COTR
                corrs = self.engine.cotr_corr_multiscale(
                    img1, img2,
                    zoom_ins=np.linspace(0.75, 0.1, 4),
                    converge_iters=1,
                    max_corrs=points.shape[0],
                    queries_a=points.astype(np.float64), force=True, areas=[1.0, 1.0]
                )

                if corrs.shape[0] < 4:
                    # Not enough matches
                    continue

                # Fing homography relating both sets of correspondences
                H, mask_H = cv2.findHomography(corrs[:, :2], corrs[:, 2:], cv2.RANSAC, 10.0)

                if mask_H.sum() < 4:
                    # Homography not found
                    continue

                # Calculate determinant
                det = abs(np.linalg.det(H))

                # Save best homography until the moment
                if det > max_det:
                    max_det = det
                    best_H = H
                    best_points = points
                    best_points_extra = points_extra
                    best_corrs = corrs
                    if det > 0.1:
                        # The homography is good and we can stop
                        break

            component_extra = {
                'component_mask': component_mask,
                'points': best_points,
                'points_extra': best_points_extra,
                'corrs': best_corrs
            }

            # Check best homography
            if max_det > 1e-5:
                # A correct homograhpy was found, transform the mask
                transformed_mask = cv2.warpPerspective(component_mask, best_H, [component_mask.shape[1], component_mask.shape[0]])

                area_ratio = np.count_nonzero(transformed_mask) / np.count_nonzero(component_mask)
                if self.verbose:
                    print(f'Area ratio: {area_ratio}')

                if area_ratio < 5:
                    # Add transformed mask
                    final_mask = np.logical_or(final_mask, transformed_mask)
                    component_extra['ok'] = True
                    component_extra['transformed_mask'] = transformed_mask
                else:
                    # If the transformation is not good, use the orignal mask
                    final_mask = np.logical_or(final_mask, component_mask)
                    component_extra['ok'] = False
                    component_extra['transformed_mask'] = component_mask

            else:
                # Use original mask
                final_mask = np.logical_or(final_mask, component_mask)
                if return_extra:
                    component_extra['ok'] = False
                    component_extra['transformed_mask'] = component_mask

            extra.append(component_extra)

        if not return_extra:
            return final_mask
        else:
            return final_mask, extra

    def _get_connected_components(self, mask, min_area=50):
        num_components, components, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), cv2.CV_32S)
        components_list = []

        # First component is the background
        for i in range(1, num_components):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                component_mask = (components == i).astype(np.uint8)
                components_list.append(component_mask)

        return components_list

    def _get_descriptive_points(self, mask, margin_thres=2, distance_thres=10, return_extra=True):
        contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,  cv2.CHAIN_APPROX_NONE)

        # Select the longest contour
        sorted_contours = sorted(contours, key=lambda c: len(c), reverse=True)
        contour = np.array(sorted_contours[0]).squeeze()

        # Select the outermost points of the contour
        outermost_idx = [
            contour[:, 0].argmin(),
            contour[:, 0].argmax(),
            contour[:, 1].argmin(),
            contour[:, 1].argmax()
        ]
        outermost_mask = np.zeros(contour.shape[0], dtype='bool')
        outermost_mask[outermost_idx] = True

        outermost_points = contour[outermost_mask]
        selected_points = outermost_points
        remaining_points = contour[~outermost_mask]

        # Sparse sampling
        i = 0
        while i < remaining_points.shape[0]:
            # Select the first point
            p = remaining_points[i]
            selected_points = np.vstack([selected_points, p])

            dist = np.linalg.norm(remaining_points[i:] - p, axis=1)
            next_idxs = np.argwhere(dist > distance_thres)

            if next_idxs.shape[0] > 0:
                # Advance to next point
                i += next_idxs[0][0]
            else:
                break

        selected_points = self._apply_margin(selected_points, margin_thres, mask.shape)
        outermost_points = self._apply_margin(outermost_points, margin_thres, mask.shape)

        extra = {
            'margin_thres': margin_thres,
            'distance_thres': distance_thres,
            'contour': contour,
            'outermost_points': outermost_points,
        }

        if not return_extra:
            return selected_points
        else:
            return selected_points, extra

    def _apply_margin(self, points, margin_thres, shape):
        # Remove points close to the border
        margin_mask = functools.reduce(np.logical_and, [
            points[:, 0] >= margin_thres,
            points[:, 0] <= shape[1] - margin_thres,
            points[:, 1] >= margin_thres,
            points[:, 1] <= shape[0] - margin_thres
        ])
        return points[margin_mask]