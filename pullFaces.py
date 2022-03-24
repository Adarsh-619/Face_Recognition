from PIL import Image
import face_recognition

image = face_recognition.load_image_file("./assets/img/groups/team1.jpg")
face_locations = face_recognition.face_locations(image)


for i, face_loc in enumerate(face_locations):
    top, right, bottom, left = face_loc

    # Printing the edges of the faces
    print(f"Face-{i+1}", "(", top, ",", right, ",", bottom, ",", left, ")")

    # Capturing and showing the image
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
