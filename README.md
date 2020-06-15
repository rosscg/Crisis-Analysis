# Crisis-Analysis
Quantitative analysis of crisis events as part of D.Phil project at Oxford.
Markdown copies are included for viewing from within GitHub.
These are considered works-in-progress.

## Research Question
The primary goal of this research is to develop methods by which to filter
social media content that is generated during crisis events for messages which
contain 'ground truth' information.

These notebooks represent research logbooks for the benefit of the author, and
therefore are considered works-in-progress which are not intended as final
pieces. For these, refer to the related publications or contact the author
directly.

## Data Source
The data for most of this work was collected using a custom Twitter streaming
app, written by the author, which collected live data during various crisis
events. The live method allowed for detection of changing metrics (such as
changes in follower networks) and collection of detailed user network data.
The software is available in
[this repository](https://github.com/rosscg/crisis-data).

Within the notebooks, data is pulled directly from the local Postgres databases
using Django syntax (via the Django `shell_plus` extension) as the data
collection software uses the Django framework to record its data.
