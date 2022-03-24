import face_recognition
from PIL import Image, ImageDraw, ImageFont

# Loading the known images and getting the face_encodings
image_of_bill = face_recognition.load_image_file(
    "./assets/img/known/Bill Gates.jpg")
face_encoding_bill = face_recognition.face_encodings(image_of_bill)[0]
# print(face_encoding_bill)

image_of_steve = face_recognition.load_image_file(
    "./assets/img/known/Steve Jobs.jpg")
face_encoding_steve = face_recognition.face_encodings(image_of_steve)[0]
# print(face_encoding_steve)


# Create array of encodings and names
known_face_encodings = [face_encoding_bill, face_encoding_steve]

known_face_names = ["Bill Gates", "Steve Jobs"]

# Load the test image to find the faces in
test_image = face_recognition.load_image_file(
    "./assets/img/groups/bill-steve.jpg")

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Setting the font size
font = ImageFont.truetype("arial.ttf", 40)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding)

    name = "Unknown Person"

    # If Match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    # Draw Box
    draw.rectangle(((left, top), (right, bottom)), width=10,
                   outline=(128, 0, 128))

    # Draw Label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 50),
                   (right, bottom)), fill=(128, 0, 128), width=11, outline=(128, 0, 128))
    draw.text((left + 50, bottom - text_height - 40),
              name, fill=(255, 255, 255, 255), font=font)


del draw

# Display image
pil_image.show()
