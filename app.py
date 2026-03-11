import cv2
from transformers import pipeline
from PIL import Image

# Load fruit classification model
classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

# Start camera
cap = cv2.VideoCapture(0)

print("Press SPACE to capture fruit")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()

    cv2.imshow("Fruit Recognition Camera", frame)

    key = cv2.waitKey(1)

    # SPACE key → capture image
    if key == 32:
        cv2.imwrite("captured_fruit.jpg", frame)
        print("Image Captured!")

        # Convert OpenCV image to PIL
        image = Image.fromarray(frame)

        # Predict fruit
        results = classifier(image)

        label = results[0]["label"]
        score = results[0]["score"]

        print(f"Prediction: {label} ({score:.2f})")

        # Show prediction on camera
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Fruit Recognition Camera", frame)
        cv2.waitKey(3000)

    # ESC key → exit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()