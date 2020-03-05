# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
from pyzbar import pyzbar
import numpy as np
import argparse
import imutils
import cv2

class Paper:
    def __init__(self, image):
        self.image = cv2.imread(image)
        self.SCORE_KEYS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}
        self.ID_KEYS = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
        self.sections = self.find_sections()
    
    def find_sections(self):
        main_sections = self.find_roi(self.image, cv2.RETR_EXTERNAL)
        result = {"questions": []}
        for s in range(len(main_sections)):
            sec_type = self.section_type(main_sections[s])
            if sec_type == "header":
                result["header"] = main_sections[s]
            elif sec_type == "question":
                result["questions"].append(main_sections[s])
                
        return result

    def section_type(self, section):
        x, y, w, h = cv2.boundingRect(section)

        ROI = self.image[y:y+h, x:x+w]

        barcodes = pyzbar.decode(ROI)
        data = barcodes[0].data.decode("utf-8")

        # TODO: header has "," questions not, or need extra data like "main main-back or additional, additional-back"
        if data in ["1023", "1024", "1025"]:
            return "question"
        else:
            return "header"

    def is_main(self):
        print("is_main")

    def is_additional(self):
        print("is_additional")

    def metadata(self):
        x, y, w, h = cv2.boundingRect(self.sections["header"])

        ROI = self.image[y:y+h, x:x+w]

        barcodes = pyzbar.decode(ROI)
        data = barcodes[0].data.decode("utf-8")

        return data
    
    def find_contours(self, edged_image, retr):
        cnts = cv2.findContours(edged_image, retr, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts

    def find_roi(self, image, retr):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 90)

        sections = []
        cnts = self.find_contours(edged,retr)
        # ensure that at least one contour was found
        if len(cnts) > 0:
            # sort the contours according to their size in
            # descending order
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            # loop over the sorted contours
            for c in cnts:
                # approximate the contour
                if cv2.contourArea(c) > 300:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    # if our approximated contour has four points,
                    # then we can assume we have found the paper
                    if len(approx) == 4:
                        sections.append(approx)
                        # break
        #cv2.drawContours(image, sections, -1, (0, 255, 0), 3)
        #cv2.imshow('Contours', image)
        #cv2.waitKey(0)
        #print(len(sections))
        return sections

    def student_id_number(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 90)

        x, y, w, h = cv2.boundingRect(self.sections["header"])
        gray_roi = gray[y:y+h, x:x+w]
        ROI = self.image[y:y+h, x:x+w]
        edged_roi = edged[y:y+h, x:x+w]
        # cv2.imshow('Section Contour', ROI)
        # cv2.waitKey(0)

        score_section = self.find_roi(ROI,cv2.RETR_TREE)[4]

        x, y, w, h = cv2.boundingRect(score_section)
        gray_roi = gray_roi[y:y+h, x:x+w]
        ROI = ROI[y:y+h, x:x+w]
        edged_roi = edged_roi[y:y+h, x:x+w]
        #cv2.imshow('Score Contour', ROI)
        #cv2.waitKey(0)

        #top = int(0.01 * ROI.shape[0])  # shape[0] = rows
        #bottom = top
        #left = int(0.01 * ROI.shape[1])  # shape[1] = cols
        #right = left

        #ROI = cv2.copyMakeBorder(ROI, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        #gray_roi = cv2.copyMakeBorder(gray_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 0))

        return self.read_box(imutils.rotate_bound(ROI, -270), imutils.rotate_bound(gray_roi, -270), 32, self.ID_KEYS)

    def total_score(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 90)

        x, y, w, h = cv2.boundingRect(self.sections["header"])
        gray_roi = gray[y:y+h, x:x+w]
        ROI = self.image[y:y+h, x:x+w]
        edged_roi = edged[y:y+h, x:x+w]
        # cv2.imshow('Section Contour', ROI)
        # cv2.waitKey(0)

        score_section = self.find_roi(ROI,cv2.RETR_TREE)[2]

        x, y, w, h = cv2.boundingRect(score_section)
        gray_roi = gray_roi[y:y+h, x:x+w]
        ROI = ROI[y:y+h, x:x+w]
        edged_roi = edged_roi[y:y+h, x:x+w]
        # cv2.imshow('Score Contour', ROI)
        # cv2.waitKey(0)

        return self.read_box(ROI, gray_roi, 29, self.SCORE_KEYS)
    
    def question_score(self, question_section):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 90)

        x, y, w, h = cv2.boundingRect(question_section)

        gray_roi = gray[y+3:y+h, x-5:x+w]
        ROI = self.image[y+3:y+h, x-5:x+w]
        edged_roi = edged[y+3:y+h, x-5:x+w]
        # cv2.imshow('ROI', ROI)
        # cv2.waitKey(0)

        # all = self.find_roi(ROI, cv2.RETR_TREE)
        # cv2.drawContours(ROI, all, -1, (0, 255, 0), 3)
        # cv2.imshow('Contours', ROI)
        # cv2.waitKey(0)

        score_section = self.find_roi(ROI, cv2.RETR_TREE)[0]

        x, y, w, h = cv2.boundingRect(score_section)
        gray_roi = gray_roi[y:y+h, x+5:x+w]
        ROI = ROI[y:y+h, x+5:x+w]
        edged_roi = edged_roi[y:y+h, x+5:x+w]
        # cv2.imshow('ROI', ROI)
        # cv2.waitKey(0)

        return self.read_box(ROI, gray_roi, 29, self.SCORE_KEYS)

    def question_scores(self):
        result = {}
        questions = self.sections["questions"]
        for q in range(len(questions)):
            x, y, w, h = cv2.boundingRect(questions[q])
            ROI = self.image[y:y+h, x:x+w]
            result[self.barcode_data(ROI)] = self.question_score(questions[q])
        return result

    def read_box(self, roi, gray_roi, circle_width, keys):
        # apply Otsu's thresholding method to binarize the warped
        # piece of paper
        thresh = cv2.threshold(gray_roi, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find contours in the thresholded image, then initialize
        # the list of contours that correspond to questions
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # cnts = [cv2.convexHull(c) for c in cnts]

       # cv2.drawContours(roi, cnts, -1, (0, 0, 255), 3)

        #cv2.imshow("questionCnts", roi)
        #cv2.waitKey(0)
        #print(len(cnts))

        questionCnts = []
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour, then use the
            # bounding box to derive the aspect ratio
            (_, _, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # in order to label the contour as a question, region
            # should be sufficiently wide, sufficiently tall, and
            # have an aspect ratio approximately equal to 1
            # print("{}, {}".format(w,h))
            # 32 additional paper
            if w >= circle_width and h >= circle_width and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)

        #print(len(questionCnts))
        #cv2.drawContours(roi, questionCnts, -1, (0, 0, 255), 3)

        #cv2.imshow("questionCnts", roi)
        #cv2.waitKey(0)
        # sort the question contours top-to-bottom, then initialize
        # the total number of correct answers
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

        # each question has 5 possible answers, to loop over the
        # question in batches of 5
        score = ""
        # print(len(questionCnts))
        for (_, i) in enumerate(np.arange(0, len(questionCnts), 10)):
            # sort the contours for the current question from
            # left to right, then initialize the index of the
            # bubbled answer
            cnts = contours.sort_contours(questionCnts[i:i + 10])[0]
            bubbled = None

            # loop over the sorted contours
            for (j, c) in enumerate(cnts):
                # construct a mask that reveals only the current
                # "bubble" for the question
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                # apply the mask to the thresholded image, then
                # count the number of non-zero pixels in the
                # bubble area
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                # if the current total has a larger number of total
                # non-zero pixels, then we are examining the currently
                # bubbled-in answer
                if bubbled is None or total > bubbled[0]: # pylint: disable=E1136
                    bubbled = (total, j)

            score += str(keys[bubbled[1]])

            # draw the outline of the bubbled
            # color = (0, 0, 255)
            # cv2.drawContours(roi, [cnts[bubbled[1]]], -1, color, 3)
        # cv2.imshow("questionCnts", roi)
        # cv2.waitKey(0)
        return score

    def barcode_data(self, image):
        barcodes = pyzbar.decode(image)
        return barcodes[0].data.decode("utf-8")
    

def main():
    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    args = vars(ap.parse_args())

    paper = Paper(args["image"])

    print("Paper Metadata:", paper.metadata())
    print("Total Score:", paper.total_score())
    print("Question Scores:", paper.question_scores())
    #print(paper.student_id_number())

if __name__ == '__main__':
    main()
