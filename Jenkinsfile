pipeline {
    agent any

    environment {
        PROJECT_DIR = "/var/lib/jenkins/workspace/HealthScope"
        IMAGE_NAME = "healthscope-web"
        GIT_REPO = "https://github.com/Jimit2304/HealthScope.git"
    }

    triggers {
        pollSCM('H/1 * * * *') // Automatically check GitHub every 1 minute
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo "📦 Pulling latest code from GitHub..."
                dir("${PROJECT_DIR}") {
                    deleteDir() // Clean old files
                    git branch: 'main', url: "${GIT_REPO}"
                }
            }
        }

        stage('Build & Deploy Containers') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                    echo "🐳 Stopping old containers..."
                    docker-compose down || true

                    echo "🧱 Rebuilding containers with latest code..."
                    docker-compose build --no-cache

                    echo "🚀 Starting updated containers..."
                    docker-compose up -d
                    '''
                }
            }
        }
    }

    post {
        success {
            echo "✅ Deployment completed with updated code!"
        }
        failure {
            echo "❌ Deployment failed. Check Jenkins logs!"
        }
    }
}
