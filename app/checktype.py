from django.core.exceptions import ValidationError
import os
def validate_file_type(value):

    try:

        ext = os.path.splitext(value.name)[1]  # [0] returns path + filename
        valid_extensions = ['.jpg', '.png', '.jpeg']
        if not ext.lower() in valid_extensions:
            raise ValidationError('Unsupported file extension.')
        filesize = value.size
        if filesize > 5242880:
            raise ValidationError("The maximum file size that can be uploaded is 5 MB")
    except Exception as e:
        return False
    return True