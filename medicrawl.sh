#!/bin/bash
CONFIGMAP_NAME="medicrawl-config"
KEY="medications"

# Get the list of medications
medications=$(kubectl get configmap $CONFIGMAP_NAME -o jsonpath="{.data.$KEY}" | tr '\n' ',')

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: medicrawl-pod
spec:
  restartPolicy: Never
  containers:
    - name: medicrawl
      image: aidan2b/adr-detection:latest
      imagePullPolicy: Always
      command: ["/bin/sh", "-c"]
      env:
        - name: MEDICATIONS
          value: "$medications"
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
      args:
        - |
          # Loop through the medications
          for medication in \$(echo \$MEDICATIONS | tr ',' ' '); do
            export MEDICATION_NAME="\$medication"

            # Print the start of processing for the medication
            echo "Starting processing for medication: \$MEDICATION_NAME"
            
            # Run the pipeline script
            # python run_pipeline.py
            
            # Copy the output files to the shared volume on initial run
            # cp -r /shiny_app/linked_data.csv /data/linked_data.csv

            # Copy the shiny app files to the shared volume on initial run
            cp -r /shiny_app/server.R /data/server.R
            # cp -r /shiny_app/ui.R /data/ui.R
            
            # Append the output files to the shared volume on subsequent runs
            tail -n +2 /shiny_app/linked_data.csv >> /data/linked_data.csv

            # Print the end of processing for the medication
            echo "Finished processing for medication: \$MEDICATION_NAME"
          done
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
