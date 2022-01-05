from django.forms import Form, ImageField


class ImgForm(Form):
    img_field = ImageField()