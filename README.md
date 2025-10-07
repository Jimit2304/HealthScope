# HealthScope - Diabetes Prediction Web Application

A machine learning-powered web application that predicts diabetes risk using health metrics and provides comprehensive health monitoring features.

## Features

- **Diabetes Risk Prediction**: ML-based prediction using health parameters
- **User Authentication**: Secure login/register system with role-based access
- **Prediction History**: Track and export prediction history
- **Admin Dashboard**: Comprehensive analytics and user management
- **Risk Factor Analysis**: Detailed health risk assessment
- **Batch Predictions**: Process multiple predictions at once
- **Data Export**: Export prediction data to CSV

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: MySQL
- **ML**: Scikit-learn (Logistic Regression)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy

## Installation

### Prerequisites
- Python 3.7+
- MySQL Server
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Jimit2304/HealthScope.git
cd HealthScope
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure MySQL**
- Create database: `diapredict`
- Update credentials in `app.py`:
```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "diapredict"
}
```

4. **Initialize database**
```bash
python setup_db.py
```

5. **Run the application**
```bash
python app.py
```

## Usage

### Access the Application
- Local: `http://localhost:5000`
- Network: `http://[your-ip]:5000`

### User Roles
- **Regular User**: Make predictions, view personal history
- **Admin** (`admin12`): Access all features, user management, analytics

### Making Predictions
Input health parameters:
- Pregnancies
- Glucose level
- Blood pressure
- Insulin level
- BMI
- Age
- Family diabetes history

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/register` | User registration |
| POST | `/api/login` | User login |
| POST | `/api/predict` | Single prediction |
| POST | `/api/batch-predict` | Batch predictions |
| GET | `/api/history` | Prediction history |
| GET | `/api/dashboard/stats` | Dashboard statistics |
| GET | `/api/export/history` | Export data to CSV |

## Model Information

- **Algorithm**: Logistic Regression with StandardScaler
- **Features**: 6 key health parameters
- **Dataset**: Diabetes dataset with preprocessing
- **Accuracy**: ~77% (varies with data)

## File Structure

```
HealthScope/
├── app.py              # Main Flask application
├── setup_db.py         # Database initialization
├── requirements.txt    # Python dependencies
├── diabetes[1].csv     # Training dataset
├── templates/          # HTML templates
│   ├── home.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── history.html
│   └── stats.html
└── static/            # CSS, JS, images
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

- **Developer**: Jimit
- **GitHub**: [@Jimit2304](https://github.com/Jimit2304)
- **Repository**: [HealthScope](https://github.com/Jimit2304/HealthScope)

---

⚠️ **Disclaimer**: This application is for educational purposes only. Always consult healthcare professionals for medical advice.