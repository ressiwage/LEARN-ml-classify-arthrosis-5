import cv2
def preprocess(path):
    img = cv2.resize(cv2.imread(path),(112,112), interpolation = cv2.INTER_AREA)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return [equalized_img/255]
