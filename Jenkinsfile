pipeline {
    agent any
    
    environment {
        DOCKER_HUB_CREDENTIALS = credentials('dockerhub-credentials')
        DOCKER_HUB_REPO = 'abdulahad2242/ml-ops-app'
        EMAIL_RECIPIENTS = 'abdulahadabbassi2@gmail.com'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    def imageTag = "${BUILD_NUMBER}"
                    def imageName = "${DOCKER_HUB_REPO}:${imageTag}"
                    
                    // Build Docker image
                    sh "docker build -t ${imageName} ."
                    sh "docker tag ${imageName} ${DOCKER_HUB_REPO}:latest"
                    
                    env.IMAGE_NAME = imageName
                }
            }
        }
        
        stage('Test Docker Image') {
            steps {
                script {
                    // Test the Docker image
                    sh """
                    docker run --rm -d --name test-container -p 5001:5000 ${env.IMAGE_NAME}
                    sleep 10
                    curl -f http://localhost:5001/health || exit 1
                    docker stop test-container
                    """
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                script {
                    // Login to Docker Hub
                    sh 'echo $DOCKER_HUB_CREDENTIALS_PSW | docker login -u $DOCKER_HUB_CREDENTIALS_USR --password-stdin'
                    
                    // Push images
                    sh "docker push ${env.IMAGE_NAME}"
                    sh "docker push ${DOCKER_HUB_REPO}:latest"
                    
                    // Cleanup
                    sh 'docker logout'
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                sh """
                docker rmi ${env.IMAGE_NAME} || true
                docker rmi ${DOCKER_HUB_REPO}:latest || true
                docker system prune -f
                """
            }
        }
    }
    
    post {
        success {
            emailext (
                subject: "✅ Jenkins Job Success: ${env.JOB_NAME} - Build ${env.BUILD_NUMBER}",
                body: """
                <h2>Jenkins Job Completed Successfully! 🎉</h2>
                <p><strong>Job:</strong> ${env.JOB_NAME}</p>
                <p><strong>Build Number:</strong> ${env.BUILD_NUMBER}</p>
                <p><strong>Docker Image:</strong> ${env.IMAGE_NAME}</p>
                <p><strong>Status:</strong> SUCCESS</p>
                <p><strong>Duration:</strong> ${currentBuild.durationString}</p>
                
                <h3>Actions Completed:</h3>
                <ul>
                    <li>✅ Docker image built successfully</li>
                    <li>✅ Image tested and validated</li>
                    <li>✅ Pushed to Docker Hub</li>
                    <li>✅ Cleanup completed</li>
                </ul>
                
                <p><strong>Console Output:</strong> <a href="${env.BUILD_URL}console">${env.BUILD_URL}console</a></p>
                <p><strong>Build Details:</strong> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></p>
                """,
                mimeType: 'text/html',
                to: env.EMAIL_RECIPIENTS
            )
        }
        failure {
            emailext (
                subject: "❌ Jenkins Job Failed: ${env.JOB_NAME} - Build ${env.BUILD_NUMBER}",
                body: """
                <h2>Jenkins Job Failed! ❌</h2>
                <p><strong>Job:</strong> ${env.JOB_NAME}</p>
                <p><strong>Build Number:</strong> ${env.BUILD_NUMBER}</p>
                <p><strong>Status:</strong> FAILED</p>
                <p><strong>Duration:</strong> ${currentBuild.durationString}</p>
                
                <p><strong>Console Output:</strong> <a href="${env.BUILD_URL}console">${env.BUILD_URL}console</a></p>
                <p><strong>Build Details:</strong> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></p>
                
                <p>Please check the logs and fix the issues.</p>
                """,
                mimeType: 'text/html',
                to: env.EMAIL_RECIPIENTS
            )
        }
    }
}