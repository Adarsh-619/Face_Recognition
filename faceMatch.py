import face_recognition

image_of_bill = face_recognition.load_image_file(
    "./assets/img/known/Bill Gates.jpg")
face_encoding_bill = face_recognition.face_encodings(image_of_bill)
# print(face_encoding_bill)

bill = face_encoding_bill[0]

unknown_image = face_recognition.load_image_file(
    "./assets/img/unknown/gates_lookalike.jpg")
face_encoding_unknown = face_recognition.face_encodings(unknown_image)
# print(face_encoding_unknown)

unknown = face_encoding_unknown[0]

# Compare Faces
results = face_recognition.compare_faces(
    [bill], unknown)

print(results)
