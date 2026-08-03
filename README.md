# Week 7 – Stress Testing, Observability & Scaling the IRIS Pipeline

**Name:** Khushi Choudhary
**IITM BS Roll Number:** 23f3002872
**Repository:** 23f3002872_MLOPS_WEEKLY_ASSIGNMENT
**Branch:** week_7

## Objective

The objective of this assignment is to validate the deployed IRIS inference API under high concurrency, observe Kubernetes autoscaling behavior, and identify bottlenecks when scaling is constrained. This extends the Week 6 CD pipeline with stress testing (`wrk`), Horizontal Pod Autoscaling (HPA), and GCP-native observability (Cloud Monitoring + Cloud Logging).

## Architecture

```
Code Push
    |
    v
CI (Tests)
    |
    v
CD (Build -> Push to Artifact Registry -> Deploy to GKE)
    |
    v
Live IRIS API (LoadBalancer)
    |
    v
Stress Testing, Observability & Scaling
    |-- wrk: High-concurrency load generation
    |-- HPA: CPU-based pod autoscaling (min:1, max:3)
    |-- GCP Cloud Monitoring: Pod/Node CPU & memory dashboards
    |-- GCP Cloud Logging: Request-level log correlation
    |-- Bottleneck Analysis: Constrained scaling comparison
```

## Technologies Used

Python, FastAPI, Docker, Kubernetes, Google Kubernetes Engine (GKE), Google Artifact Registry, GitHub Actions, Google Cloud Platform, GCP Cloud Monitoring, GCP Cloud Logging, `wrk`, Horizontal Pod Autoscaler (HPA).

## Infrastructure Configuration

| Component | Value |
|---|---|
| GCP Project | `mlops-project-500020` |
| Artifact Registry | `us-central1-docker.pkg.dev/mlops-project-500020/iris-repo` |
| GKE Cluster | `iris-cluster` (zone: `us-central1-a`) |
| Node Pool | 2 x `e2-small`, 20GB `pd-standard` disks |
| Deployment resource requests | `cpu: 100m`, `memory: 256Mi` |
| Deployment resource limits | `cpu: 500m`, `memory: 512Mi` |
| Service | `iris-api-service` (LoadBalancer) |

## Task 1: Extend CI/CD Workflow with Stress Testing

The existing GitHub Actions workflow (`cd.yml`) was extended with three new steps that run automatically after a successful deployment:

1. **Install wrk** on the runner (`apt-get install wrk`).
2. **Get External IP** — polls `kubectl get svc` until the LoadBalancer IP is assigned.
3. **Wait for API readiness** — polls `/health` until it returns `200`, preventing a race condition where `wrk` fires before the LoadBalancer backend is fully healthy.
4. **Run Stress Test (wrk)** — executes `wrk -t4 -c1000 -d30s -s post.lua` against the live `/predict` endpoint.

The workflow triggers automatically on every push to the `week_7` branch. All steps — checkout, authenticate, build, push, deploy, rollout verification, and stress test — run end-to-end in a single CI run (confirmed passing in Actions run "Build and Deploy to GKE #7").

`post.lua` (committed to repo root) sends the required JSON payload since `/predict` only accepts POST:

```lua
wrk.method = "POST"
wrk.body   = '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
wrk.headers["Content-Type"] = "application/json"
```

## Task 2: Simulate High-Concurrency Traffic with wrk

Command used:

```bash
wrk -t4 -c1000 -d30s -s post.lua http://<EXTERNAL_IP>/predict
```

**Result (with HPA active, 3 replicas available across a 2-node cluster):**

```
Running 30s test @ http://136.64.59.204/predict
  4 threads and 1000 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     1.21s   488.38ms   2.00s    61.74%
    Req/Sec     69.27     61.51   330.00    81.05%
  3580 requests in 30.08s, 517.42KB read
  Socket errors: connect 0, read 0, write 0, timeout 538
Requests/sec:    119.03
Transfer/sec:     17.20KB
```

538 of 3580 requests (~15%) timed out under 1000 concurrent connections — the API remained largely responsive but showed clear latency pressure at this concurrency level.

## Task 3: Configure Horizontal Pod Autoscaler

`k8s/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iris-api
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
```

During the Task 2 load test, HPA scaled the deployment from 1 to 3 replicas as CPU utilization exceeded target:

```
NAME           REFERENCE             TARGETS       MINPODS   MAXPODS   REPLICAS
iris-api-hpa   Deployment/iris-api   cpu: 497%/50%   1         3         3
iris-api-hpa   Deployment/iris-api   cpu: 306%/50%   1         3         3
iris-api-hpa   Deployment/iris-api   cpu: 257%/50%   1         3         3
```

```
NAME                        READY   STATUS    RESTARTS   AGE
iris-api-664b87df8b-5slv9   1/1     Running   0          72s
iris-api-664b87df8b-797qv   1/1     Running   0          14h
iris-api-664b87df8b-g5mcq   1/1     Running   0          72s
```

**Important infrastructure note:** the cluster initially ran on a single `e2-small` node, which only had ~940m allocatable CPU after GKE system daemonset overhead — insufficient for 3 replicas to actually schedule (2 of 3 pods stuck `Pending` with `Insufficient cpu`). The node pool was resized to 2 nodes so that all 3 HPA-requested replicas could genuinely reach `Running`, giving a valid comparison baseline for Task 5.

## Task 4: Monitor with GCP Cloud Monitoring & Cloud Logging

**Cloud Monitoring** (GKE Compute Resources - Workload View, filtered to the `iris-api` deployment) showed a clear CPU spike per pod during the load test — CPU request utilization jumped to ~300% and limit utilization to ~60% across the 3 running replicas, directly correlating with the HPA scale-up event.

**Cloud Logging** (Logs Explorer), filtered with:

```
resource.type="k8s_container"
resource.labels.cluster_name="iris-cluster"
resource.labels.pod_name=~"iris-api"
```

showed 43,670+ `POST /predict HTTP/1.1" 200 OK` entries during the test window, originating from distinct internal source IPs (`10.128.0.29`, `10.24.0.1`) corresponding to different pods/nodes — confirming load was distributed across all running replicas rather than concentrated on one.

## Task 5: Observe Bottlenecks Under Constrained Scaling

HPA was patched to a single-replica ceiling:

```bash
kubectl patch hpa iris-api-hpa -p '{"spec":{"maxReplicas":1}}'
```

Load was then increased to 2000 concurrent connections:

```bash
wrk -t4 -c2000 -d30s -s post.lua http://136.64.59.204/predict
```

**Result:**

```
Running 30s test @ http://136.64.59.204/predict
  4 threads and 2000 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     0.00us    0.00us   0.00us    -nan%
    Req/Sec     48.09     43.25   210.00    80.93%
  3665 requests in 30.04s, 529.71KB read
  Socket errors: connect 0, read 65, write 0, timeout 3665
Requests/sec:    122.00
Transfer/sec:     17.63KB
```

### Comparison Table

| Metric | Task 3 (max 3 replicas, 1000 conns) | Task 5 (max 1 replica, 2000 conns) |
|---|---|---|
| Total requests | 3580 | 3665 |
| Timeouts | 538 (~15%) | 3665 (**100%**) |
| Requests/sec | 119.03 | 122.00 |
| Measurable latency | avg 1.21s, max 2.00s | unmeasurable (0 completed within window) |
| Replicas actually running | 3 | 1 (pinned, confirmed via `kubectl get hpa`) |
| Peak CPU utilization | 497%/50% target | 499%/50% target |

### Bottleneck Analysis

CPU is the first resource to saturate in both scenarios (utilization far exceeds the 50% target), but the effect is dramatically different depending on available replica headroom. With 3 replicas, the extra capacity absorbs enough of the load that ~85% of requests still complete successfully. With scaling capped at 1 replica, the single pod's CPU limit (`500m`) becomes a hard ceiling — every request queues behind it and none complete within `wrk`'s timeout window.

Notably, Cloud Logging showed the single pod **did** log `200 OK` for the requests it processed during this window (2,992 requests logged in a 30-second sample) — meaning the pod wasn't crashing or erroring server-side. The bottleneck was response **latency**, not application failure: under CPU saturation, response times exceeded `wrk`'s client-side timeout before the (successful) response could be returned, so `wrk` reports these as timeouts even though the server eventually completed them. This distinguishes a capacity/latency bottleneck from an application-level failure — the fix is more compute headroom (replicas and/or CPU), not a code fix.

## Learning Outcomes

Through this assignment, I learned: high-concurrency load testing with `wrk`, configuring Kubernetes HPA with resource-based scaling triggers, the relationship between pod resource requests/limits and node allocatable capacity, diagnosing `Insufficient cpu` scheduling failures, using GCP Cloud Monitoring dashboards to correlate autoscaling events with resource usage, using Cloud Logging to distinguish client-side timeouts from server-side failures, and integrating stress testing directly into a CI/CD pipeline.

## Conclusion

This assignment validated the IRIS API's behavior under high-concurrency load and demonstrated Kubernetes HPA scaling pods from 1 to 3 in response to CPU pressure. Cloud Monitoring and Cloud Logging provided direct visibility into per-pod resource consumption and request distribution during the test. Constraining `maxReplicas` to 1 and increasing load to 2000 connections produced a complete breakdown in successful request completion (0% vs ~85% success), clearly identifying CPU capacity — not application logic — as the primary bottleneck, and showing that Kubernetes autoscaling is necessary but not sufficient without adequate underlying node capacity to host the scaled replicas.