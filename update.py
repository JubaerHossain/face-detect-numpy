import os
import numpy as np
from flask import Flask, request, jsonify
import cv2
from mtcnn import MTCNN
from datetime import datetime
import mysql.connector
from concurrent.futures import ThreadPoolExecutor
import pika
import tensorflow as tf

app = Flask(__name__)

# Initialize MTCNN face detector
detector = MTCNN()

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'your_database'
}

# RabbitMQ configuration
rabbitmq_config = {
    'host': 'localhost',
    'port': 5672,
    'username': 'your_username',
    'password': 'your_password',
    'queue': 'face_recognition_queue'
}

# Connect to the MySQL database
db_connection = mysql.connector.connect(**db_config)
db_cursor = db_connection.cursor()

# Path to the directory for storing employee images
employee_images_directory = "employee_images/"

# Face recognition model
face_recognition_model = None
employee_labels = []

# Pose validation thresholds
left_pose_thresholds = {
    'shoulder_angle': 30,
    'hip_angle': 30
}

right_pose_thresholds = {
    'shoulder_angle': 30,
    'hip_angle': 30
}

straight_pose_thresholds = {
    'shoulder_angle': 30,
    'hip_angle': 30
}

# Initialize RabbitMQ connection
rabbitmq_connection = pika.BlockingConnection(pika.ConnectionParameters(
    host=rabbitmq_config['host'],
    port=rabbitmq_config['port'],
    credentials=pika.PlainCredentials(rabbitmq_config['username'], rabbitmq_config['password'])
))
rabbitmq_channel = rabbitmq_connection.channel()
rabbitmq_channel.queue_declare(queue=rabbitmq_config['queue'])

def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def preprocess_image(image):
    # Resize the image to the desired input shape
    image = cv2.resize(image, (64, 64))

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    image = image / 255.0

    # Add a channel dimension to the image
    image = np.expand_dims(image, axis=-1)

    return image

def perform_face_recognition(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Detect faces in the image using MTCNN
    faces = detector.detect_faces(image)

    # Perform face recognition for each detected face
    for face in faces:
        x, y, width, height = face['box']
        face_image = image[y:y+height, x:x+width]

        # Preprocess the face image
        preprocessed_face = preprocess_image(face_image)

        # Reshape the preprocessed face to match the input shape of the model
        preprocessed_face = np.reshape(preprocessed_face, (1, 64, 64, 1))

        # Perform face recognition using the trained model
        predictions = face_recognition_model.predict(preprocessed_face)

        # Get the predicted label
        predicted_label = np.argmax(predictions)

        # Get the employee ID corresponding to the predicted label
        employee_id = employee_labels[predicted_label]

        # Perform attendance marking for the detected employee
        mark_attendance_in_database(employee_id)

    # Delete the temporary image file
    os.remove(image_path)

def mark_attendance_in_database(employee_id):
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Insert attendance record into the database
    insert_query = "INSERT INTO attendance (employee_id, timestamp) VALUES (%s, %s)"
    attendance_data = (employee_id, timestamp)

    db_cursor.execute(insert_query, attendance_data)
    db_connection.commit()

def validate_pose(image, pose_label):
    # Perform pose estimation on the image using OpenPose
    pose_estimation_model = ...

    # Get the pose angles from the pose estimation results
    shoulder_angle = ...
    hip_angle = ...

    # Get the pose validation thresholds based on the pose label
    if pose_label == 'left':
        thresholds = left_pose_thresholds
    elif pose_label == 'right':
        thresholds = right_pose_thresholds
    elif pose_label == 'straight':
        thresholds = straight_pose_thresholds
    else:
        return False

    # Check if the pose angles are within the valid range
    if abs(shoulder_angle) <= thresholds['shoulder_angle'] and abs(hip_angle) <= thresholds['hip_angle']:
        return True

    return False

@app.route('/store-employee-image', methods=['POST'])
def store_employee_image():
    if 'image' not in request.files or 'employee_id' not in request.form:
        return jsonify({'error': 'Invalid request'}), 400

    # Read the image file from the request
    image_file = request.files['image']
    employee_id = request.form['employee_id']

    # Create the employee directory if it doesn't exist
    employee_directory = os.path.join(employee_images_directory, employee_id)
    if not os.path.exists(employee_directory):
        os.makedirs(employee_directory)

    # Save the image file
    image_path = os.path.join(employee_directory, "face.jpg")
    image_file.save(image_path)

    return jsonify({'result': 'Employee image stored successfully'}), 200

@app.route('/train-model', methods=['POST'])
def train_model():
    # Get all employee directories
    employee_directories = [d for d in os.listdir(employee_images_directory) if os.path.isdir(os.path.join(employee_images_directory, d))]

    # Load and preprocess employee images
    employee_images = []
    employee_labels.clear()

    for i, employee_dir in enumerate(employee_directories):
        images = []
        for image_file in os.listdir(os.path.join(employee_images_directory, employee_dir)):
            image_path = os.path.join(employee_images_directory, employee_dir, image_file)
            image = cv2.imread(image_path)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        if len(images) > 0:
            employee_images.append(np.array(images))
            employee_labels.append(employee_dir)

    if len(employee_images) == 0:
        return jsonify({'error': 'No employee images found'}), 400

    # Convert employee labels to numerical values
    label_mapping = {label: i for i, label in enumerate(employee_labels)}
    employee_labels_numeric = [label_mapping[label] for label in employee_labels]

    # Preprocess employee images
    employee_images_preprocessed = []
    for images in employee_images:
        preprocessed_images = []
        for image in images:
            preprocessed_image = cv2.resize(image, (64, 64))
            preprocessed_images.append(preprocessed_image)
        employee_images_preprocessed.append(np.array(preprocessed_images))

    # Convert the preprocessed images to tensors
    employee_images_preprocessed = [tf.convert_to_tensor(images, dtype=tf.float32) for images in employee_images_preprocessed]

    # Train the face recognition model
    global face_recognition_model
    face_recognition_model = create_model((64, 64, 1))
    face_recognition_model.fit(np.concatenate(employee_images_preprocessed), np.array(employee_labels_numeric), epochs=10)

    return jsonify({'result': 'Model trained successfully'}), 200

@app.route('/attendance', methods=['POST'])
def process_attendance_request():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    # Read the image file from the request
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Validate the left pose
    if validate_pose(image, 'left'):
        # Validate the right pose
        if validate_pose(image, 'right'):
            # Validate the straight pose
            if validate_pose(image, 'straight'):
                # Generate a unique filename for the image
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                image_filename = f"temp_image_{timestamp}.jpg"

                # Save the image to a temporary file
                image_path = os.path.join("temp_images", image_filename)
                cv2.imwrite(image_path, image)

                # Send the image path to the face recognition queue
                rabbitmq_channel.basic_publish(
                    exchange='',
                    routing_key=rabbitmq_config['queue'],
                    body=image_path.encode()
                )

                return jsonify({'result': 'Attendance request received'}), 200

    return jsonify({'error': 'Pose validation failed'}), 400

def start_background_worker():
    def callback(ch, method, properties, body):
        # Perform face recognition in the background
        perform_face_recognition(body.decode())

    rabbitmq_channel.basic_consume(
        queue=rabbitmq_config['queue'],
        on_message_callback=callback,
        auto_ack=True
    )

    # Start consuming messages in the background
    rabbitmq_channel.start_consuming()

if __name__ == '__main__':
    # Create the temporary images directory if it doesn't exist
    temp_images_directory = "temp_images"
    if not os.path.exists(temp_images_directory):
        os.makedirs(temp_images_directory)

    # Start the background worker in a separate thread
    with ThreadPoolExecutor() as executor:
        executor.submit(start_background_worker)

    # Run the Flask application
    app.run(threaded=True)
