minikube start --driver=docker
minikube dashboard

minikube image load forecast-service:latest
minikube image load data-ingestion:latest

kubectl create secret generic ingestion-secret

kubectl apply -f pv.yaml

kubectl apply -f forecast-deployment.yaml
kubectl apply -f forecast-service.yaml
kubectl apply -f data-ingestion-deployment.yaml
kubectl apply -f forecast-hpa.yaml

kubectl delete deployment forecast-deployment
kubectl delete deployment data-ingestion-deployment

kubectl get pods
kubectl get deployments

kubectl logs -f <new-pod-name>

kubectl port-forward svc/forecast-service 8000:80

kubectl create job --from=cronjob/data-ingestion manual-ingestion
kubectl logs -f job/manual-ingestion

kubectl rollout restart deployment/forecast-deployment


# ingress
minikube addons enable ingress
minikube addons list
kubectl apply -f ingress.yaml
kubectl get ingress
kubectl describe ingress renewable-ingress
minikube ip