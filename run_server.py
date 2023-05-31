from flask import Flask, request, jsonify
import pickle

import sys

sys.path.append('./')
app = Flask(__name__)

def load_models():
    with open('checkpoint/maxent_M100_iter5_acc0.9.pkl', 'rb') as f:
        maxent_model = pickle.load(f)

    # XXX_model = XXXModel()

    return {
        'maxent': maxent_model
        #'XXX_model': XXX_model
    }

model_dict = load_models()

@app.route('/classify', methods=['GET'])
def classify():
    data = request.get_json()
    text = data['text']
    model_type = data['model_type']

    if model_type == 'maxent':
        result = model_dict['maxent'].predict_texts([text]).tolist()[0]
    elif model_type == 'model2':
        # 调用另一个模型进行分类
        # ...
        result = 'Model 2 classification result'
    elif model_type == 'model3':
        # 调用另一个模型进行分类
        # ...
        result = 'Model 3 classification result'
    else:
        return jsonify({'error': 'Invalid model type'})

    return jsonify({
        'result': result  # int 0 or 1
        # 'prob': prob # 需要返回概率吗
    })

if __name__ == '__main__':
    app.run()
