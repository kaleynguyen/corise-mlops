# Week 1
## ML definition in terms of Task, Experience and Performance
### 1. Task: What problem would I like to solve?
Scope and project setup as an optimization problem with respect to : 'Optimize business metric X with bounds on cost, latency and other business metric Y'

### 2. Experience: What data should I collect? 
> No data and invest in annotation
> Datasets are noisy and stored in different location
> The real world data evolves and leads to potential model drift
> Data can be biased which impacts the model performance on some cohort. 

### 3. Performance: The performance metrics to evaluate the quality of production ML systems. This includes both the model performance and model evaluation. 

#### 3.1. Model Performance
There are 3 types of performance hurdles: training + dev dataset => test dataset => product/business goal.

1. Start simple: ensure that feature transformation and training pipelines have no bugs. This can be done
by writing unit tests and train on small dataset. 
2. Establish a baseline performance. The delta diff = human perf - baseline perf can suggest how much a model  
can improve. 
3. Establish a well-defined process for error analysis. Datalog for DAG of data pipelines? 

#### 3.2. Model Evaluation
* Aggregate performance: all MSEs, accuracy or AUC are important but far from sufficient.
* Cohort-level performance: defining cohort based on the product.
* Mapping model metrics to product metrics: how much money is saved by detecting fraud that was previously undetected?
* Qualitative evaluation:  Error analysis and qualitative evaluation can be helpful for identifying the root causes of  individual instances of model failures.

## ML as a new paradigm to build software
* Silent failures: models failing to produce the correct output without any explicit warnings.
* Blackbox: not interpretable
* Drifts: train model represents the state of the data as a snapshot in time but fail to represent the dynamic 
changing of data => cause models to regress over time. 

# Week 2

