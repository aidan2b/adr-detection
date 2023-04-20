#!/bin/bash
CONFIGMAP_NAME="medications-configmap"
KEY="medications"

for medication in $(kubectl get configmap $CONFIGMAP_NAME -o jsonpath="{.data.$KEY}" | tr '\n' ' ')
do
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: pod
metadata:
  name: medicrawl-pod-$medication
spec:
  template:
restartPolicy: Never
containers:
- name: medicrawl
    image: aidan2b/adr-detection:test
    command: ["/bin/sh", "-c"]
    env:
    - name: MEDICATION_NAME
        value: "$medication"
    - name: REDDIT_CLIENT_ID
        valueFrom:
        secretKeyRef:
            name: adr-detection-secrets
            key: REDDIT_CLIENT_ID
    - name: REDDIT_CLIENT_SECRET
        valueFrom:
        secretKeyRef:
            name: adr-detection-secrets
            key: REDDIT_CLIENT_SECRET
    - name: SHINYAPPS_TOKEN
        valueFrom:
        secretKeyRef:
            name: adr-detection-secrets
            key: SHINYAPPS_TOKEN
    - name: SHINYAPPS_SECRET
        valueFrom:
        secretKeyRef:
            name: adr-detection-secrets
            key: SHINYAPPS_SECRET
    args:
    - |
        # Run the pipeline script
        python run_pipeline.py

        cp -r /shiny_app/faers.csv /data/faers.csv
        tail -n +2 /shiny_app/linked_data.csv >> /data/linked_data.csv

        # Deploy to shinyapps.io
        # export PATH=$PATH:/usr/bin/R
        # R -e "rsconnect::setAccountInfo(name = 'aidan2b', token = Sys.getenv('SHINYAPPS_TOKEN'), secret = Sys.getenv('SHINYAPPS_SECRET'))"
        # R -e "rsconnect::deployApp(appDir = 'data/', appName = 'adr-detection', account = 'aidan2b')"
    resources:
    limits:
        memory: 12Gi
        cpu: 2
        nvidia.com/gpu: 1
    requests:
        memory: 12Gi
        cpu: 2
        nvidia.com/gpu: 1
    volumeMounts:
    - mountPath: /data
        name: adr-detection-pvc
volumes:
- name: adr-detection-pvc
    persistentVolumeClaim:
    claimName: adr-detection-pvc
affinity:
nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
        - matchExpressions:
            - key: nvidia.com/gpu.product
            operator: In
            values:
                - NVIDIA-GeForce-RTX-3090
                - Tesla-T4

EOF
done
