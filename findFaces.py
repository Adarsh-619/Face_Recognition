import face_recognition

image = face_recognition.load_image_file("./assets/img/groups/team2.jpg")

face_locations = face_recognition.face_locations(image)

# Array of coordinates of each face
print(face_locations)


# No of people in the image
print(f"There are {len(face_locations)} people in the image.")
