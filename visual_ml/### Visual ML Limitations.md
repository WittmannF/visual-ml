### Visual ML Issues and Limitations

- The input set of features shoudl be a Pandas Dataframe
- When 1D the column should be declared as a string, not a list ('col_name' instead of ['col_name'])
- The target variable has to be binary and either True/False or 1/0. 
    - Support to regression and multiclassification can be implemented in the future
- The training and testing set can't be visualized as different sets.

- Histogram colors are sometimes inverted
- The Python code could be more elegant and robust

- add a list of features parameter
- when 1D, len(col) is wrong
- 

### Issue

When feat_list is only one feature, there's an error:
Example:
```
columns_list = ['A','B','C','D','E']
X, y = make_classification(n_features=n_feats, random_state=42)
X = pd.DataFrame(X, columns=columns_list)
clf = RF(random_state=42).fit(X,y)
vml.decision_boundary_grid(clf, X, y, feat_list='A')
```

Will throw the following error:

```
Traceback (most recent call last):
  File "/Users/wittmann/projects/visual-ml/visual_ml/test_visual_ml.py", line 87, in <module>
    main()
  File "/Users/wittmann/projects/visual-ml/visual_ml/test_visual_ml.py", line 83, in main
    test_decision_boundary_grid(n_feats=5, feat_list='A')
  File "/Users/wittmann/projects/visual-ml/visual_ml/test_visual_ml.py", line 20, in test_decision_boundary_grid
    vml.decision_boundary_grid(clf, X, y, feat_list=feat_list)#, figsize=(10,10))
  File "/Users/wittmann/projects/visual-ml/visual_ml/visualml.py", line 57, in decision_boundary_grid
    ax_i = ax[i_y][i_x] # Row first, which is the Y axis
TypeError: 'AxesSubplot' object does not support indexing
[Finished in 1.8s with exit code 1]
[shell_cmd: python -u "/Users/wittmann/projects/visual-ml/visual_ml/test_visual_ml.py"]
[dir: /Users/wittmann/projects/visual-ml/visual_ml]
[path: /usr/bin:/bin:/usr/sbin:/sbin]
```