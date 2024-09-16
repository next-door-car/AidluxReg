import cv2
import numpy as np
from PIL import Image
def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class ImageProcessor:
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.image = cv2.imread(image_path)
        self.mask = cv2.imread(mask_path)
        self.processed_image = None

    def image_angles(self, image):
        kerne2 = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kerne2)
        edges = cv2.Canny(closing, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        angles1 = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles1.append(angle1)

        if angles1:
            average_angle1 = np.mean(angles1)
            print(f'Average Angle1: {average_angle1}')
        else:
            average_angle1 = 0
            print("No lines detected with average_angle1.")

        return average_angle1
    
    def calculate_black_density(self, image, threshold=50):
        black_pixels = np.sum(image < threshold)
        total_pixels = image.size
        return black_pixels / total_pixels
    
    def show(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        image_array = np.array(self.image)
        mask_array = np.array(self.mask)
        result_array = image_array - mask_array

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])
        hsv_image = cv2.cvtColor(result_array, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        combined_mask = cv2.bitwise_or(yellow_mask, green_mask)
        result_array[combined_mask > 0] = [255, 255, 255]
        result_array[(result_array > 100).all(axis=-1)] = [255, 255, 255]

        result_image = Image.fromarray(result_array)
        result_cv2 = np.array(result_image)
        median_blur_image = cv2.medianBlur(result_cv2, 5)
        median_blur_image = cv2.GaussianBlur(median_blur_image, (5, 5), 0)
        image = cv2.cvtColor(median_blur_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        average_angle1 = self.image_angles(opening)
        median_image = cv2.medianBlur(opening, 5)
        kerne3 = np.ones((40, 40), np.uint8)
        dilated_image = cv2.dilate(median_image, kerne3, iterations=1)
        edges = cv2.Canny(dilated_image, 70, 200, apertureSize=5, L2gradient=False)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        line_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
        angles = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

        if angles:
            average_angle = np.mean(angles)
            print(f'Average Angle: {average_angle}')
        else:
            average_angle = 0
            print("No lines detected with average_angle.")
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_rect = None
        best_box = None

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            area = rect[1][0] * rect[1][1]
            if area > max_area:
                max_area = area
                best_rect = rect
                best_box = box

        print("最大矩形面积：", max_area)

        if best_rect is not None:
            current_width, current_height = best_rect[1]
            print("当前矩形宽高：", current_width, current_height)
            width1 = current_width
            height1 = current_height
            if height1 > width1:
                width1, height1 = height1, width1
            if width1 < 1024:
                width1 = 1024
            if height1 < 384:
                height1 = 384

            expanded_rect = ((best_rect[0][0], best_rect[0][1]), (current_width, current_height), best_rect[2])
            expanded_box = cv2.boxPoints(expanded_rect)
            expanded_box = np.intp(expanded_box)

            cv2.drawContours(line_image, [expanded_box], 0, (255, 0, 0), 2)
            show(line_image)

            # 提取旋转外接矩形
            center = best_rect[0]
            size = (int(width1), int(height1))
            weight_factor = 0.98  # 设定权重因子，范围可以根据需要调整
            if average_angle1 != 0:
                angle = weight_factor * average_angle1 + (1 - weight_factor) * average_angle
            else:
                angle = average_angle

            # 获取旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rows, cols = self.image.shape[:2]

            # 增加边界填充
            padX = int(rows * 0.5)
            padY = int(cols * 0.5)
            padded_image = cv2.copyMakeBorder(self.image, padY, padY, padX, padX, cv2.BORDER_REPLICATE)

            # 更新中心点坐标
            new_center = (center[0] + padX, center[1] + padY)

            # 重新计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(new_center, angle, 1.0)
            rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (cols + 2 * padX, rows + 2 * padY), flags=cv2.INTER_CUBIC)

            # 提取旋转后的图像
            extracted_image = cv2.getRectSubPix(rotated_image, size, new_center)
            show(extracted_image)


            gray_extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2GRAY)
            width1 = int(width1)
            if width1 % 2 == 1:
                width1 += 1
            left_half = gray_extracted_image[:, :width1 // 2]
            right_half = gray_extracted_image[:, width1 // 2:]
            left_density = self.calculate_black_density(left_half)
            right_density = self.calculate_black_density(right_half)

            if left_density > right_density:
                extracted_image = cv2.rotate(extracted_image, cv2.ROTATE_180)

            cv2.imwrite('0.png', extracted_image)
            self.processed_image = extracted_image
        else:
            print("No contours found.")

if __name__ == '__main__':
    road1 = r'D:\python_project\AidLux\PICTURE\refer\1.png'
    road2 = r'D:\python_project\AidLux\PICTURE\mask\10.png'
    processor = ImageProcessor(road1, road2)
    processor.process_image()
