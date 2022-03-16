<p>The sequence number in selected_index.txt is the number of the 5 promoters we selected to display the DeepLIFT method, which corresponds to the number of the promoters in the test set CNN_test_leaf.tsv and CNN_test_proto.tsv.<p>

Running steps:
1. Run create_fasta.py to generate the selected_fasta.fa file;
2. Modify the path of each file in deeplift_plot.py and run deeplift_plot.py;
3. To view the DeepLIFT graph or DeepLIFT scores of other sequences, just modify the sequence number in selected_index.txt and repeat steps 1 and 2. 
