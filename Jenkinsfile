pipeline {
    agent any

    environment {
        // 🔧 Path where your project lives
        PROJECT_DIR = "/var/lib/jenkins/workspace/HealthScope"


        // 🐳 Docker image name
        IMAGE_NAME = "healthscope-web"

        // 🌐 Your GitHub repo
        GIT_REPO = "https://github.com/Jimit2304/HealthScope.git"

        // 🏷️ Branch name
        BRANCH_NAME = "main"
    }

    stages {

        stage('Prepare Workspace') {
            steps {
                sh '''
                echo "📂 Preparing workspace..."

                # Create directory if it doesn't exist
                mkdir -p ${PROJECT_DIR}

                cd ${PROJECT_DIR}

                # If repo not cloned yet, clone it
                if [ ! -d ".git" ]; then
                    echo "🔄 Cloning repository for the first time..."
                    git clone ${GIT_REPO} .
                else
                    echo "🔁 Updating existing repository..."
                    git config --global --add safe.directory ${PROJECT_DIR}
                    git reset --hard
                    git clean -fd
                    git pull origin ${BRANCH_NAME} || true
                fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                    echo "🐳 Building fresh Docker image..."
                    docker build --no-cache -t ${IMAGE_NAME}:latest .
                    '''
                }
            }
        }

        stage('Stop Old Containers') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                    echo "🧹 Stopping and removing old containers..."
                    docker-compose down || true
                    '''
                }
            }
        }

        stage('Start New Containers') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                    echo "🚀 Starting updated containers..."
                    docker-compose up -d
                    '''
                }
            }
        }
    }

    post {
        success {
            echo "✅ Deployment successful! HealthScope is live with the latest code."
        }
        failure {
            echo "❌ Pipeline failed. Check Jenkins logs for details."
        }
    }
}
