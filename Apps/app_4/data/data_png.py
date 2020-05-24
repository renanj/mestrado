import io
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np




def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64

def generate_thumbnail(image_b64):
    return html.Div([
        html.A([
            html.Img(
                src="data:image/png;base64, " + image_b64,
                style = {
                    'height': '10%',
                    'width': '10%',
                    'float': 'left',
                    'max-width': '235px',
                    'position': 'relative',
                    'padding-top': 5,
                    'padding-right': 5
                }
            )
        ],),
    ])







df = pd.read_csv('mnist_3000_input.csv')

for i in range(df.iloc[:,:].shape[0]):
  _image = Image.open(BytesIO(base64.b64decode(numpy_to_b64(df.iloc[i,:].values.reshape(28, 28).astype(np.float64)))))
  _image.save(str(i) + ".png", 'png')
