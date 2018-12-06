# cs7180_team_project
### Assignment5[80/100]
- Health check is wrongly written, currently you just return string healthy which will never result in an exception even if your service is actually unhealthy due to some issue. So health function should invoke a trained model's predict.
- Can use an integer data type instead of number since you only need integer values
