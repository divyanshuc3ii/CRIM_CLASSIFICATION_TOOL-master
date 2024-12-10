# Crime Classification Machine Learning Web Application

## Overview
This is a full-stack web application for crime classification using machine learning, built with Django (backend) and React (frontend).

## Project Structure
```
crime_classification_project/
├── backend/             # Django backend
│   ├── ml_model/        # Machine learning model files
│   ├── classification/  # Django app for ML predictions
│   └── crime_classifier/# Django project settings
│
└── frontend/            # React frontend
    ├── src/             # React component source code
    └── public/          # Static assets
```

## Prerequisites
- Python 3.8+
- Node.js 14+
- pip
- npm

## Installation

### Backend Setup
1. Navigate to backend directory
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Configure Django
```bash
python manage.py migrate
python manage.py runserver
```

### Frontend Setup
1. Navigate to frontend directory
```bash
cd frontend
npm install
npm start
```

## Environment Configuration
- Update `SECRET_KEY` in `backend/crime_classifier/settings.py`
- Adjust `CORS_ALLOWED_ORIGINS` for production deployment

## Model Placement
- Place model files in `backend/ml_model/BEST/`
- Ensure `label_mapping_WO_CV.json` is in `backend/ml_model/`

## Features
- Text-based crime classification
- Machine learning inference
- Responsive web interface
- Detailed prediction probabilities

## Technologies
- Backend: Django, PyTorch
- Frontend: React, Tailwind CSS
- ML Libraries: Transformers, SpaCy, NLTK

## Deployment
- Backend: Django server
- Frontend: React development server
- Recommended: Use Gunicorn/uWSGI for production backend

## Troubleshooting
- Ensure all dependencies are installed
- Check CORS settings
- Verify model file paths

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License
Apache License 2.0
