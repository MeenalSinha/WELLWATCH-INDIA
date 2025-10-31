# WellWatch India - Deployment Guide

## 🚀 Deployment Options

### Option 1: Local Development
1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Run Streamlit: `streamlit run streamlit_app.py`
4. Access at: http://localhost:8501

### Option 2: Google Colab (Demo)
1. Upload notebook to Google Drive
2. Open in Colab
3. Run all cells
4. Use ngrok for public URL

### Option 3: Cloud Deployment (Production)

#### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click
4. Free tier available

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Deploy
heroku create wellwatch-india
git push heroku main
```

#### AWS EC2
```bash
# On EC2 instance
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
nohup streamlit run streamlit_app.py --server.port 8501 &
```

### Option 4: Mobile App Integration

#### React Native Integration
```javascript
// API call from mobile app
const assessRisk = async (patientData) => {
  const response = await fetch('https://api.wellwatch.in/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patientData)
  });
  return await response.json();
};
```

## 🔒 Security Considerations
- Use HTTPS for all API calls
- Implement authentication for production
- Encrypt patient data at rest
- Follow HIPAA/local healthcare data regulations
- Regular security audits

## 📊 Monitoring & Maintenance
- Track prediction accuracy over time
- Monitor API response times
- Regular model retraining (quarterly)
- User feedback collection
- A/B testing for model improvements

## 🔄 Update Process
1. Retrain model with new data
2. Validate performance metrics
3. Run unit tests
4. Deploy to staging
5. User acceptance testing
6. Production deployment
7. Monitor for 48 hours

## 📞 Support
For deployment support: devops@wellwatch.in
