# Converting bridge2ai redcap to reproschema

This repository contains the version-controlled implementation of the Bridge2AI ReproSchema, converted from REDCap data dictionaries, using the [`redcap2reproschema`](https://github.com/ReproNim/reproschema-py#redcap2reproschema-usage) tool

## Conversion Process

For each REDCap data dictionary version, we did the following

### 1. Removal of Previous Elements

All existing protocols, activities, and items from the previous version were deleted to ensure a clean starting point for the new version. This step was critical to clearly document which elements were removed between versions

### 2. Convert to reproschema format

`reproschema-py` requires Python 3.10+ and can be installed directly through `pip`:

```
pip install reproschema
```

Or, you can istall the developer mode through GitHub:
```
git clone https://github.com/ReproNim/reproschema-py.git
cd reproschema-py
pip install -e .
```
Convert the data dictionary
```
reproschema redcap2reproschema [b2ai-redcap-data-dictionary.csv] [redcap2rs.ymal]
```

### 3. Git add and commit

Add and commit the newly converted reproschema
```
git add .
git commit -m "converted b2ai redcap data dictionary version xx to reproschema"
```

### 4. Version tagging

After committing the changes, each version was tagged to mark the release point in the repository's history. For example:
```
git tag -a 1.0.0 -m "redcap data dictionary version 1.0 to reproschema"
```

### 5. Git push

Push the current tag to remote
```
# push the tag
git push origin <tag_name>
# push the content
git push 
# if there is a conflict, you can use --force
git push -f
```
