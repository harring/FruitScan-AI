from django.test import TestCase
from .models import ImageData

class ImageDataTestCase(TestCase):
    def setUp(self):
        # Example image data (bytes)
        example_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...'
        
        # Create an ImageData instance
        ImageData.objects.create(label='A', image_data=example_image_data)

    def test_image_data_content(self):
        # ImageData objects have the correct label and image data.
        image_data_instance = ImageData.objects.get(label='A')
        
        # Check if the label is correct
        self.assertEqual(image_data_instance.label, 'A')
        
        # Check if the image data is correct (this is a simplistic check)
        self.assertTrue(image_data_instance.image_data.startswith(b'\x89PNG'))
