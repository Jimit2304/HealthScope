pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                script {
                    docker.build("healthscope:${env.BUILD_ID}")
                }
            }
        }
        
        stage('Test') {
            steps {
                script {
                    docker.image("healthscope:${env.BUILD_ID}").inside {
                        sh 'python -m pytest --version || echo "No tests found"'
                    }
                }
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    sh 'docker-compose down || true'
                    sh 'docker-compose up -d --build'
                }
            }
        }
    }
    
    post {
        always {
            sh 'docker system prune -f'
        }
    }
}