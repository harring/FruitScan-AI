from django.test import TestCase
from .models import CustomUser, FruitClassification, ImageData, TestImageData, UploadedImage, UserImage, user_directory_path

class ImageDataTestCase(TestCase):
    def setUp(self):
        # Example image data (bytes)
        example_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...'
        
        # Create an ImageData instance
        ImageData.objects.create(label='1', image_data=example_image_data)

    def test_image_data_content(self):
        # ImageData objects have the correct label and image data.
        image_data_instance = ImageData.objects.get(label='1')
        
        # Check if the label is correct
        self.assertEqual(image_data_instance.label, '1')
        
        # Check if the image data is correct (this is a simplistic check)
        self.assertTrue(image_data_instance.image_data.startswith(b'\x89PNG'))

class UploadedImageTestCase(TestCase):
    def setUp(self):
        # Create an UploadedImage instance and save it
        self.image = UploadedImage.objects.create(image="image.jpg")

    def test_uploaded_image_content(self):
        # UploadedImage objects have the correct image path.
        image = UploadedImage.objects.get(id=self.image.id)

        # Get the expected relative image path
        expected_image_path = 'image.jpg'

        # Check if the image path is correct
        self.assertEqual(image.image.name, expected_image_path)

class TestImageDataTestCase(TestCase):
    def setUp(self):
        # Example image data (bytes)
        example_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...'

        # Create a TestImageData instance
        TestImageData.objects.create(label='B', image_data=example_image_data)

    def test_test_image_data_content(self):
        # TestImageData objects have the correct label and image data.
        image_data_instance = TestImageData.objects.get(label='B')

        # Check if the label is correct
        self.assertEqual(image_data_instance.label, 'B')

        # Check if the image data is correct (this is a simplistic check)
        self.assertTrue(image_data_instance.image_data.startswith(b'\x89PNG'))

class CustomUserTestCase(TestCase):
    def test_custom_user_str(self):
        # Check the string representation of a CustomUser object
        user = CustomUser.objects.create(username='testuser')
        self.assertEqual(str(user), 'testuser')

class UserImageTestCase(TestCase):
    def setUp(self):
        # Create a CustomUser instance
        self.user = CustomUser.objects.create(username='testuser')

        # Create a UserImage instance related to the user
        self.image_instance = UserImage.objects.create(
            user=self.user,
            image="media/images/user_1/image.jpg",
            pred="prediction"
        )

    def test_user_image_content(self):
        image_instance = UserImage.objects.get(id=self.image_instance.id)
        expected_image_path = f'media/images/user_{image_instance.user.id}/image.jpg'

        # Check if the image path is correct
        self.assertEqual(image_instance.image.name, expected_image_path)

        # Check if the user is correct
        self.assertEqual(image_instance.user.username, 'testuser')

        # Check if the prediction is correct
        self.assertEqual(image_instance.pred, "prediction")
