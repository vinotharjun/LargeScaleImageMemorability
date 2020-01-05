#### helper function that is used in experiments

def show_grid(n_row,n_col):

  i=900
  _ ,fig = plt.subplots(n_row, n_col, figsize=(12,12))
#   print(fig)
  fig = fig.flatten()
#   print(fig)
  for f in fig:
#     print(i,f)
    image_path=os.path.join("/content/lamem/images/"+str(dataset_train.iloc[i]["X"]))
    mem_value = os.path.join(str(dataset_train.iloc[i]["y"]))
    f.imshow(io.imread(image_path))
    f.title.set_text(str(mem_value))
#     f.title(str(aseth_value))
    i=i+1
#show_grid(3,3)
