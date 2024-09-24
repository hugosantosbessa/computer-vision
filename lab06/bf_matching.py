import cv2
import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt


def match_images(im1,im2,kp1,kp2,des1,des2, n_show = 20, use_knn=False):
	# create BFMatcher object
	bf = cv.BFMatcher()

	if not use_knn:
		# Match descriptors.
		matches = bf.match(des1, des2)

		# Sort them in the order of their distance.
		matches = sorted(matches, key=lambda x: x.distance)

		# Draw first matches.
		img_match = cv.drawMatches(img1, kp1, img2, kp2, matches[:n_show], None,
								   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	else:
		matches = bf.knnMatch(des1, des2, k=2)
		# Apply ratio test
		good = []
		for m, n in matches:
			if m.distance < 0.75 * n.distance:
				good.append([m])

		img_match = cv.drawMatchesKnn(img1, kp1, img2, kp2, good[:n_show], None,
									  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	return img_match


def bf_orb(im1,im2, n_show = 20, use_knn=False):
	orb = cv.ORB_create()

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(im1,None)
	kp2, des2 = orb.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)





def bf_sift(im1,im2, n_show=20, use_knn=False):
	# Initiate SIFT detector
	sift = cv.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(im1,None)
	kp2, des2 = sift.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)






############


img1 = cv.imread("../images/antartica.jpg")
img2 = cv.imread("../images/antartica_lata.jpg")


#SIFT
im_sift = bf_sift(img1,img2,use_knn=True)


#ORB
im_orb = bf_orb(img1,img2,use_knn=True)



plt.subplot(121).set_ylabel("SIFT"), plt.imshow(im_sift,'gray') #imagem original
plt.subplot(122).set_ylabel("ORB"), plt.imshow(im_orb,'gray') #imagem original


plt.show()

