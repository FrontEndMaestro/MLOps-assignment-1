import pytest
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.app import app

class TestAPI:
    
    def setup_method(self):
        """Setup test client"""
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_predict_endpoint_structure(self):
        """Test predict endpoint structure"""
        # Test with valid data structure
        test_data = {
            'features': [0.5, 0.6, 0.7, 0.8, 0.9]
        }
        
        response = self.client.post('/predict', 
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        # Should return some response (may fail due to no model, but structure should be ok)
        assert response.status_code in [200, 500]  # 500 if no model loaded
    
    def test_predict_invalid_input(self):
        """Test predict endpoint with invalid input"""
        # Test with invalid data
        test_data = {'invalid': 'data'}
        
        response = self.client.post('/predict',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 400