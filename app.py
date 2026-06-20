from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from heat_transfer import calculate_temperature_distribution
import os

app = Flask(__name__, static_folder='static')
CORS(app)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()

        inner_temp = float(data.get('inner_temp', 150.0))
        insulation_thickness = float(data.get('insulation_thickness', 0.05))
        env_temp = float(data.get('env_temp', 25.0))

        if insulation_thickness <= 0:
            return jsonify({'error': '保温层厚度必须大于0'}), 400

        result = calculate_temperature_distribution(
            inner_temp=inner_temp,
            insulation_thickness=insulation_thickness,
            env_temp=env_temp
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5001)
