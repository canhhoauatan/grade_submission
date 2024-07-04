import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
def resize(img, height=800):
    """ Resize image to given height """
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

def fourCornersSort(pts):
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)

    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])

def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt

def find_and_number_squares(image):
    # Load image, grayscale, median blur, sharpen image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 140, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    close = cv2.dilate(close, kernel, iterations=1)
    close = cv2.erode(close, kernel, iterations=1)

    # Find contours and filter using threshold area
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 800
    max_area = 5000
    image_number = 0
    squares = []

    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                squares.append(approx)

    for i, square in enumerate(squares):
        cv2.drawContours(image, [square], -1, ((106, 90, 205)), -1)
        M = cv2.moments(square)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [square], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        if mean_val < 128:
            cv2.putText(image, f"{i + 1} {""}", (cX - 10, cY + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def find_rectangular(image, large_y_tolerance=1000, small_y_tolerance=150):
    # Load image, grayscale, adaptive threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, ((255,0,255)), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Draw rectangles and save the cropped images in a list
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rectangles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 100000:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255,140,0), 4)
            ROI = image[y:y + h, x:x + w]
            rectangles.append((x, y, ROI))

    # Group rectangles by large y coordinate with large tolerance
    rectangles.sort(key=lambda r: r[1])
    large_rows = []
    current_large_row = []

    for rect in rectangles:
        if not current_large_row:
            current_large_row.append(rect)
        else:
            if abs(rect[1] - current_large_row[0][1]) <= large_y_tolerance:
                current_large_row.append(rect)
            else:
                large_rows.append(current_large_row)
                current_large_row = [rect]

    if current_large_row:
        large_rows.append(current_large_row)

    # Sort each large row by small y coordinate and x coordinate
    sorted_images = []
    for large_row in large_rows:
        small_rows = []
        current_small_row = []

        large_row.sort(key=lambda r: r[1])
        for rect in large_row:
            if not current_small_row:
                current_small_row.append(rect)
            else:
                if abs(rect[1] - current_small_row[0][1]) <= small_y_tolerance:
                    current_small_row.append(rect)
                else:
                    small_rows.append(current_small_row)
                    current_small_row = [rect]

        if current_small_row:
            small_rows.append(current_small_row)

        for small_row in small_rows:
            small_row.sort(key=lambda r: r[0])
            sorted_images.extend([rect[2] for rect in small_row])

    final_images = []
    # Process large images
    for img in sorted_images:
        if img.shape[1] > 1500:
            part_width = img.shape[1] // 6
            img = img[130:, :]
            for i in range(6):
                part = img[:, i * part_width: (i + 1) * part_width]
                final_images.append(part)
        elif img.shape[0] < 500:
            # Chia ảnh thành 2 phần, phần đầu tiên có chiều rộng là 100
            # img = img[120:, :]
            if img.shape[1] > 100:
                part1 = img[:, :240]
                part2 = img[:, 240:]
                final_images.append(part1)
                final_images.append(part2)
            else:
                final_images.append(img)  # Nếu chiều rộng ảnh nhỏ hơn 100, thêm cả ảnh vào
        else:
            final_images.append(img)

    return final_images












def determine_answers(answers, width, height, proportions, num_questions):
    # Sắp xếp các câu trả lời theo tọa độ y trước, rồi đến tọa độ x
    sorted_answers = sorted(answers, key=lambda x: (x[1], x[0]))
    # Tính chiều cao của mỗi câu hỏi
    question_height = height / num_questions

    if num_questions == 10:
        answer_sections = ['A', 'B', 'C', 'D']
    elif num_questions == 4:
        answer_sections = ['T', 'F']

    # Khởi tạo danh sách các câu hỏi với giá trị None
    questions = ['?'] * num_questions

    for (x, y) in sorted_answers:
        # Xác định chỉ số câu hỏi dựa trên tọa độ y
        question_index = int(y // question_height)
        if question_index < num_questions:
            cumulative_width = 0
            for i, proportion in enumerate(proportions):
                cumulative_width += width * proportion
                # Xác định phần trả lời dựa trên tọa độ x
                if x <= cumulative_width:
                    if questions[question_index] == '?':
                        questions[question_index] = answer_sections[i]
                    else:
                        questions[question_index] += answer_sections[i]
                    break

    # Nối các phần trả lời thành chuỗi kết quả
    return ''.join(questions)


def get_bubble_contours(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
                               param1=100, param2=30, minRadius=16, maxRadius=23)
    num_circles = 0
    num_filled_circle = 0
    answers = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        bubble_count = 1  # Initialize counter for bubble numbering

        proportions = [0.30, 0.25, 0.23, 0.22]
        col_ranges = [(image.shape[1] * sum(proportions[:i]), image.shape[1] * sum(proportions[:i+1])) for i in range(4)]
        column_bubbles = [[] for _ in range(4)]  # List to store circles in each column

        for (x, y, r) in circles:
            # Determine the column for the bubble
            for col_index, (col_start, col_end) in enumerate(col_ranges):
                if col_start <= x < col_end:
                    column_bubbles[col_index].append((x, y, r))
                    break  # Exit inner loop after finding the column

        # Loop through each column and its bubbles
        for col_index, bubbles in enumerate(column_bubbles):
            # Sort bubbles based on their y-coordinate (top to bottom)
            sorted_bubbles = sorted(bubbles, key=lambda b: b[1])
            for (x, y, r) in sorted_bubbles:
                # Draw the center of the circle
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

                # Create a small region of interest (ROI) around the circle
                roi = gray[y-r:y+r, x-r:x+r]
                
                # Apply a binary threshold to the ROI
                _, binary_roi = cv2.threshold(roi, 160, 255, cv2.THRESH_BINARY)

                # Calculate the number of white pixels in the binary ROI
                num_white_pixels = cv2.countNonZero(binary_roi)
                cv2.putText(image, str(bubble_count), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)  # Draw number with appropriate offset
                bubble_count += 1
                
                # If the number of white pixels is less than half the area of the circle, consider it filled
                if num_white_pixels < (np.pi * r * r) / 2:
                    cv2.circle(image, (x, y), r, ((0, 255, 129)), -1)
                    answers.append((x, y))
                    num_filled_circle += 1
                else:
                    cv2.circle(image, (x, y), r, (234, 123, 206), 3)
                num_circles += 1

    print(f"Circles: {num_circles} ({num_filled_circle})")

    proportions = []
    num_questions = 0
    final_answers = ""

    if num_circles > 41:
        filled_bubble_positions = [-1, -1, -1, -1]  # Initialize with -1 for 4 columns
        for col_index, bubbles in enumerate(column_bubbles):
            for (x, y, r) in bubbles:
                if (x, y) in answers:
                    # Find the bubble number (1-based index) within the column
                    bubble_number = sorted(bubbles, key=lambda b: b[1]).index((x, y, r)) + 1
                    filled_bubble_positions[col_index] = bubble_number
                    break  # Move to the next column once the filled bubble is found

        col_values = [
            "-0123456789",  # Cột 1 có 11 ô
            ",0123456789",  # Cột 2 có 11 ô
            ",0123456789",  # Cột 3 có 11 ô
            "0123456789"    # Cột 4 có 10 ô
        ]

        filled_values = []
        for col_index, bubble_position in enumerate(filled_bubble_positions):
            if bubble_position != -1:
                filled_values.append(col_values[col_index][bubble_position - 1])
            else:
                filled_values.append('?')  # Placeholder for columns without a filled bubble

        final_answers = ''.join(filled_values)
        final_answers = list(final_answers)

        if final_answers[1] == ',':
            final_answers[2] = '0' if final_answers[2] == '?' else final_answers[2]
            final_answers[3] = '0' if final_answers[3] == '?' else final_answers[3]
        elif final_answers[2] == ',':
            final_answers[3] = '0' if final_answers[3] == '?' else final_answers[3]

        final_answers = ''.join(final_answers)
        print("Filled values:", final_answers)

    elif num_circles == 40:
        num_questions = 10
        proportions = [0.30, 0.25, 0.23, 0.22]
        final_answers = determine_answers(answers, image.shape[1], image.shape[0], proportions, num_questions)
    elif num_circles == 8:
        num_questions = 4
        proportions = [0.55, 0.45]
        final_answers = determine_answers(answers, image.shape[1], image.shape[0], proportions, num_questions)
    else:
        final_answers = "hơi lạ, ô tròn nhận dạng sai?"

    print("Extracted answers: ", final_answers)

    return image, final_answers


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load image and convert it from BGR to RGB
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    # Resize and convert to grayscale
    img = cv2.cvtColor(resize(image), cv2.COLOR_BGR2GRAY)

    # Bilateral filter preserves edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Create black and white image based on adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    # Median filter clears small details
    img = cv2.medianBlur(img, 11)

    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    edges = cv2.Canny(img, 200, 250)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding contour of biggest rectangle
    # Otherwise return corners of original image
    # Don't forget on our 5px border!
    height = edges.shape[0]
    width = edges.shape[1]
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    # Page fill at least half of image, then saving max area found
    maxAreaFound = MAX_COUNTOUR_AREA * 0.5

    pageContour = np.array([[5, 5], [5, height-5], [width-5, height-5], [width-5, 5]])

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        # Page area must be bigger than maxAreaFound 
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):

            maxAreaFound = cv2.contourArea(approx)
            pageContour = approx

    # Result in pageContour (numpy array of 4 points):

    # Sort and offset corners
    pageContour = fourCornersSort(pageContour[:, 0])
    pageContour = contourOffset(pageContour, (-5, -5))

    # Recalculate to original scale - start Points
    sPoints = pageContour.dot(image.shape[0] / 800)

    # Using Euclidean distance
    # Calculate maximum height (maximal length of vertical edges) and width
    height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                 np.linalg.norm(sPoints[2] - sPoints[3]))
    width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                 np.linalg.norm(sPoints[3] - sPoints[0]))

    # Create target points
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)

    # Warping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints) 
    newImage = cv2.warpPerspective(image, M, (int(width), int(height)))

    # Resize the new image to A4 size (210mm x 297mm, approximately 2480 x 3508 pixels at 300 DPI)
    A4_SIZE = (2480, 3508)
    newImageA4 = cv2.resize(newImage, A4_SIZE)

    # Convert colors back to BGR
    newImageBGR = cv2.cvtColor(newImageA4, cv2.COLOR_RGB2BGR)
    crop = newImageBGR[1100:5000, 0:3000]

    image = find_and_number_squares(crop)

    rectangles = find_rectangular(image)

    answer_image_blocks = []
    part1_answer = ""
    part2_answer = ""
    part3_answer = ""
    for i, rect in enumerate(rectangles):
        circled_image, key = get_bubble_contours(rect)
        if i < 4:
            part1_answer += key.strip('?')
        elif i >= 4 and i < 12:
            part2_answer += key.strip('?')
        else:
            part3_answer += key.strip()
        
        answer_image_blocks.append(circled_image)

    print("Part_1:", part1_answer)
    print("Part_2:", part2_answer)
    print("Part_3:", part3_answer)

    final_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_image.png')
    cv2.imwrite(final_image_path, newImageBGR)

    return jsonify({
        'Part_1': part1_answer,
        'Part_2': part2_answer,
        'Part_3': part3_answer,
        'processed_image_path': final_image_path
    }), 200

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def send_processed_file(filename=''):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
