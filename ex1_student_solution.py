"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])
DEBUG = True
if DEBUG == True:
    import scipy
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    def load_data(is_perfect_matches=True):
        # Read the data:
        src_img = mpimg.imread('src.jpg')
        dst_img = mpimg.imread('dst.jpg')
        if is_perfect_matches:
            # loading perfect matches
            matches = scipy.io.loadmat('matches_perfect')
        else:
            # matching points and some outliers
            matches = scipy.io.loadmat('matches')
        match_p_dst = matches['match_p_dst'].astype(float)
        match_p_src = matches['match_p_src'].astype(float)
        return src_img, dst_img, match_p_src, match_p_dst


    src_img, dst_img, match_p_src, match_p_dst = load_data()



class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """

        num_of_points = match_p_dst.shape[1]
        if num_of_points != match_p_src.shape[1]:
            print("Issue with points' array size")
            return None
        A = np.zeros((2 * num_of_points, 9))
        for match in range(num_of_points):
            x = match_p_src[0][match]
            y = match_p_src[1][match]
            x_bar = match_p_dst[0][match]
            y_bar = match_p_dst[1][match]
            A[match*2] = [x, y, 1, 0, 0, 0, -x_bar*x, -x_bar*y, -x_bar]
            A[match*2+1] = [0, 0, 0, x, y, 1, -y_bar*x, -y_bar*y, -y_bar]

        # perform SVD
        u, s, vh = np.linalg.svd(A)
        v = vh.T
        homography_vector = v[:, -1]  # get the e.v with the smallest lambda
        homography = homography_vector.reshape((3, 3))
        return homography/homography[2,2]

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        new_image = np.zeros(dst_image_shape, dtype='uint8')
        for i in range(src_image.shape[0]):
            for j in range(src_image.shape[1]):
                new_pixel_coord = np.matmul(homography, [j, i, 1])
                new_pixel_coord_norm = new_pixel_coord/new_pixel_coord[-1]
                new_i = int(new_pixel_coord_norm[1])
                new_j = int(new_pixel_coord_norm[0])
                if 0 <= new_i < dst_image_shape[0] and 0 <= new_j < dst_image_shape[1]:
                    new_image[new_i, new_j] = src_image[i, j]
        return new_image


    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # (1) create meshgrid
        rows_idx = np.arange(0,src_image.shape[0],1)
        cols_idx = np.arange(0, src_image.shape[1], 1)
        new_image_rows, new_image_cols = np.meshgrid(rows_idx, cols_idx)
        # (2) Generate matrix with the coordinates
        src_image_homo_coord = np.ones((3, new_image_rows.size))
        src_image_homo_coord[1, :] = new_image_rows.flatten()
        src_image_homo_coord[0, :] = new_image_cols.flatten()
        # (3) Transform & norm
        new_image_home_coord = np.matmul(homography, src_image_homo_coord)
        new_image_home_coord[0] = new_image_home_coord[0]/new_image_home_coord[2]
        new_image_home_coord[1] = new_image_home_coord[1] / new_image_home_coord[2]
        # (4) convert to int and create a mask
        new_image_home_coord = new_image_home_coord.astype('int')
        src_image_homo_coord = src_image_homo_coord.astype('int')
        cols_mask = np.ma.masked_inside(new_image_home_coord[0], 0, dst_image_shape[1]-1)
        rows_mask = np.ma.masked_inside(new_image_home_coord[1], 0, dst_image_shape[0]-1)
        coord_mask = rows_mask.mask & cols_mask.mask
        # (5) Plant the pixels
        new_image = np.zeros(dst_image_shape, dtype='uint8')
        new_image[new_image_home_coord[1][coord_mask], new_image_home_coord[0][coord_mask]] = src_image[src_image_homo_coord[1][coord_mask],src_image_homo_coord[0][coord_mask]]
        return new_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        match_p_src = np.array([list(pt[:]) for pt in match_p_src.T], dtype='float32')
        match_p_dst = np.array([list(pt[:]) for pt in match_p_dst.T], dtype='float32')
        # compute homography
        match_p_src_homo = np.vstack([match_p_src.T, np.ones((1, match_p_src.shape[0]))])
        forward_map = np.matmul(homography, match_p_src_homo)
        forward_map[0] = forward_map[0] / forward_map[2]
        forward_map[1] = forward_map[1] / forward_map[2]
        forward_map = forward_map[0:2].T.astype(int)

        diff_vec = forward_map-match_p_dst
        distance_mapped_dst = np.sqrt(np.power(diff_vec[:, 0], 2) + np.power(diff_vec[:, 1], 2))
        # fit_percent is the probability that a point will be considered as inlier (distance<err)
        fit_percent = np.sum(distance_mapped_dst <= max_err) / match_p_src.shape[0]
        inlier_dist = distance_mapped_dst[distance_mapped_dst <= max_err]  # the distances of the inlier points
        # dist_mse is the MSE of the distances of the inliers (the mean squared error between the mapped to the dst)
        dist_mse = np.mean(inlier_dist)

        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        Points = np.vstack([match_p_src, np.ones((1, match_p_src.shape[1]))])
        mapped_src = np.matmul(homography, Points)
        mapped_src = mapped_src / mapped_src[-1, :]
        mapped_src = mapped_src[:-1, :]

        dists = (mapped_src[1, :] - match_p_dst[1, :]) ** 2 + (mapped_src[0, :] - match_p_dst[0, :]) ** 2
        dists = np.sqrt(dists)

        inliers = dists <= max_err

        return match_p_src[:,inliers], match_p_dst[:,inliers]

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        # w = inliers_percent
        # # t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        # p = 0.99
        # # the minimal probability of points which meets with the model
        # d = 0.5
        # # number of points sufficient to compute the model
        # n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        # k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography

        w = inliers_percent
        p = 0.99
        n = 4

        k = np.ceil(np.log(1 - p) / np.log(1 - w ** n)).astype(int)

        best_H = np.zeros((3, 3))
        best_mse = np.inf

        for i in range(k):

            # step 1 - draw a minimal random set of numbers.
            random_indxs = np.random.permutation(np.arange(match_p_src.shape[1]))[:n]

            # step 2 - fit model based on minimal number of points
            H = self.compute_homography_naive(
                match_p_src[:, random_indxs],
                match_p_dst[:, random_indxs]
            )

            # step 3 - decide if all other points are inliers/outliers
            fit_percent, mse = self.test_homography(H, match_p_src, match_p_dst, max_err)

            # step 4 - if the number of inliers is greater than d
            if fit_percent >= inliers_percent:
                all_inliers_src, all_inliers_dst = self.meet_the_model_points(H, match_p_src, match_p_dst, max_err)

                temp_H = self.compute_homography_naive(
                    all_inliers_src,
                    all_inliers_dst
                )

                fit_percent, mse = self.test_homography(H, match_p_src, match_p_dst, max_err)

                if mse <= best_mse:
                    best_mse = mse
                    best_H = temp_H

        return best_H

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        rows_idx = np.linspace(0,dst_image_shape[0]-1, dst_image_shape[0]) #np.arange(0,dst_image_shape[0],1)
        cols_idx = np.linspace(0,dst_image_shape[1]-1, dst_image_shape[1]) #np.arange(0, dst_image_shape[1], 1)
        dst_image_rows, dst_image_cols = np.meshgrid(cols_idx, rows_idx)
        # (2) Generate matrix with the coordinates
        dst_image_homo_coord = np.ones((3, dst_image_rows.size))
        dst_image_homo_coord[0, :] = dst_image_rows.flatten()
        dst_image_homo_coord[1, :] = dst_image_cols.flatten()
        # (3) Transform & norm
        new_src_image_home_coord = np.matmul(backward_projective_homography, dst_image_homo_coord)
        new_src_image_home_coord[0] = new_src_image_home_coord[0]/new_src_image_home_coord[2]
        new_src_image_home_coord[1] = new_src_image_home_coord[1] / new_src_image_home_coord[2]
        new_src_image_home_coord = new_src_image_home_coord.astype('int')

        # (4) create meshgrid of src image
        cols_mask = np.ma.masked_inside(new_src_image_home_coord[0], 0, src_image.shape[
            1]-1)
        rows_mask = np.ma.masked_inside(new_src_image_home_coord[1], 0, src_image.shape[0]-1)
        coord_mask = rows_mask.mask & cols_mask.mask


        src_image_backward_flatten = np.zeros((dst_image_cols.size,3))
        src_image_backward_flatten[coord_mask] = src_image[new_src_image_home_coord[1][coord_mask], new_src_image_home_coord[0][coord_mask]]

        src_image_mesh = (new_src_image_home_coord[1][coord_mask], new_src_image_home_coord[0][coord_mask])
        backward_warp = np.zeros(dst_image_shape, dtype='uint8')
        # (5)
        for color in range(src_image.shape[2]):
            temp = griddata(src_image_mesh, src_image_backward_flatten[:,color][coord_mask] ,(new_src_image_home_coord[1],new_src_image_home_coord[0]), method='cubic')
            backward_warp[:,:,color] = temp.reshape((dst_image_shape[0],dst_image_shape[1])).astype(np.uint8)

        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        pass

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # (1) forward homographyy
        homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        (pan_n_rows, pan_n_cols, pads) = self.find_panorama_shape(src_image, dst_image, homography)

        # (2) backward homography
        back_homography_matrix = self.compute_homography(match_p_dst, match_p_src, inliers_percent, max_err)
        # (3) add translation
        back_homography_matrix_with_translation = self.add_translation_to_backward_homography(back_homography_matrix, pads.pad_left, pads.pad_up)
        # (4) backward mapping
        src_image_wrap = self.compute_backward_mapping(back_homography_matrix_with_translation, src_image, (pan_n_rows,pan_n_cols,3))

        # (5) panorama with dst
        img_panorama = np.zeros((pan_n_rows, pan_n_cols, 3),dtype=np.uint8)
        img_panorama[pads.pad_up:pads.pad_up+dst_image.shape[0],pan_n_cols-pads.pad_right-dst_image.shape[1]: pan_n_cols-pads.pad_right,:] = dst_image[:,:,:]

        # (6) place the backward warped image
        for i in range(src_image_wrap.shape[0]):
            for j in range(src_image_wrap.shape[1]):
                if not img_panorama[i,j].any():
                    img_panorama[i, j] = src_image_wrap[i,j]

        return np.clip(img_panorama, 0, 255).astype(np.uint8)



if __name__ == '__main__':
    Solution.compute_homography_naive
    Solution.compute_backward_mapping(
            backward_projective_homography = Solution.compute_homography_naive(match_p_src,
                                                                               match_p_dst),
            src_image=src_img,
            dst_image_shape=(1088, 1452, 3))

