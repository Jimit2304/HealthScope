
pipeline {
    agent any

    environment {
        PROJECT_DIR = "/var/lib/jenkins/workspace/HealthScope"
        IMAGE_NAME = "healthscope-web"
        GIT_REPO = "https://github.com/Jimit2304/HealthScope.git"
    }

    triggers {
        pollSCM('H/5 * * * *') // Check GitHub every 5 minutes for changes
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo "📦 Checking out latest code..."
                git branch: 'main', url: "${GIT_REPO}"
            }
        }

        stage('Build Docker Image') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                    echo "🐳 Building Docker image..."
                    docker build --no-cache -t ${IMAGE_NAME}:latest .
                    '''
                }
            }
        }

        stage('Deploy Containers') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                    echo "🧹 Cleaning up old containers..."
                    docker-compose down || true

                    echo "🚀 Starting new containers..."
                    docker-compose up -d
                    '''
                }
            }
        }
    }

    post {
        success {
            echo "✅ Auto-deployment successful!"
        }
        failure {
            echo "❌ Deployment failed. Check Jenkins logs!"
        }
    }
}