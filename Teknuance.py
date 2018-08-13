
# coding: utf-8

# In[46]:


# Teknuance Task 


# In[47]:


# Importing the required header files
import numpy as np
import sklearn.cluster
import distance


# In[48]:


# Sample Word List on which we can test our algorithm
# words = "Deana, Diane, Dionne, Gerald, Irina, Lisette, Minna, Nicki, Ricki, Clair, Jani, Jason, Jc, Kimi, Lang, Marcus, Maxima, Randi, Raul, Destiny, Kellye, Marylin, Mercedes, Sterling, Verline, Elenor, Glenn, Gwenda, Armandina, Augustina, Ahmed, Estella, Milissa, Shiela, Thresa, Wynell, Autumn, Haydee, Laureen, Lauren, Albertha, Alberto, Robert, Ammie, Doreen, Eura, Josef, Lore, Lori, Porter".split(", ") 


# In[49]:


# Table Names
tnames=['Punjab_National_Bank_Customer', 'State_Bank_of_India_Customer', 'Axis_Bank_Customer', 'Dhanalaxmi_Bank_Customer', 'Union_Bank_Customer', 'HDFC_Bank_Customer', 'ICIC_Bank_Customer', 'Canara_Bank_Customer', 'Bank_of_Baroda_Customer', 'Yes_Bank_Customer', 'Kotak_Mahindra_Bank_Customer', 'IDFC_Bank_Customer', 'DCB_Bank_Customer', 'Bandhan_Bank_Customer', 'Punjab_National_Bank_Account', 'State_Bank_of_India_Account', 'Axis_Bank_Account', 'Dhanalaxmi_Bank_Account', 'Union_Bank_Account', 'HDFC_Bank_Account', 'ICIC_Bank_Account', 'Canara_Bank_Account', 'Bank_of_Baroda_Account', 'Yes_Bank_Account', 'Kotak_Mahindra_Bank_Account', 'IDFC_Bank_Account', 'DCB_Bank_Account', 'Bandhan_Bank_Account', 'Punjab_National_Bank_Branch', 'State_Bank_of_India_Branch', 'Axis_Bank_Branch', 'Dhanalaxmi_Bank_Branch', 'Union_Bank_Branch', 'HDFC_Bank_Branch', 'ICIC_Bank_Branch', 'Canara_Bank_Branch', 'Bank_of_Baroda_Branch', 'Yes_Bank_Branch', 'Kotak_Mahindra_Bank_Branch', 'IDFC_Bank_Branch', 'DCB_Bank_Branch', 'Bandhan_Bank_Branch', 'Punjab_National_Bank_Loan', 'State_Bank_of_India_Loan', 'Axis_Bank_Loan', 'Dhanalaxmi_Bank_Loan', 'Union_Bank_Loan', 'HDFC_Bank_Loan', 'ICIC_Bank_Loan', 'Canara_Bank_Loan', 'Bank_of_Baroda_Loan', 'Yes_Bank_Loan', 'Kotak_Mahindra_Bank_Loan', 'IDFC_Bank_Loan', 'DCB_Bank_Loan', 'Bandhan_Bank_Loan', 'Punjab_National_Bank_Employee', 'State_Bank_of_India_Employee', 'Axis_Bank_Employee', 'Dhanalaxmi_Bank_Employee', 'Union_Bank_Employee', 'HDFC_Bank_Employee', 'ICIC_Bank_Employee', 'Canara_Bank_Employee', 'Bank_of_Baroda_Employee', 'Yes_Bank_Employee', 'Kotak_Mahindra_Bank_Employee', 'IDFC_Bank_Employee', 'DCB_Bank_Employee', 'Bandhan_Bank_Employee']


# In[50]:


# Splitting the Table Names to get the words in the table name as a list
tnames1=[]
for i in tnames:
    tnames1.append(i.split('_'))
print(tnames1)


# In[51]:


# To store the last word in the table names
last=[]
for i in tnames1:
    last.append(i[-1])
    
print(last)
l=len(last)
print(len(last))


# In[52]:


# Dictionay to store the elements of the last array with their frequencies
dicto = {x:last.count(x) for x in last}
print(dicto)


# In[53]:


# Choosing elements of last array which occur more than 20% of the time
ans=[]
for key in dicto:
    if dicto[key]>=0.15*l:
        ans.append(key)
print(ans)


# In[54]:


# Removing Words from the end of table name if they have occured more than 15% of the time
for i in tnames1:
    for j in ans:
        if j in i:
            i.remove(j)
print(tnames1)


# In[55]:


# Creating the list of table names back after pre processing
words=[]
for i in tnames1:
    words.append(" ".join(i))
print(words)


# In[56]:


#So that indexing with a list will work
words = np.asarray(words) 


# In[57]:


# Insted of using normal Unsupervised Clustering Algorithms such as K-Means,etc. where we have to pick the 
# number of clusters through methods such as Elbow Method, etc. , I searched the net and found the 
# Affinity propogation Model which is also used as a unsupervised Clustering Algorithm in a Research Paper.
# It is genrally used for Text based Clustering.


# In[58]:


# Creating the Distance Matrix to be fed into the Affinity propoagation Model
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])


# In[59]:


# I tried various distance metrics such as Jaccard Similarity, Levenshtein distance, etc. but found that Levenshtien produced the best results.


# In[60]:


# Creating the Model
affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)


# In[61]:


# Fitting the Model
affprop.fit(lev_similarity)


# In[62]:


# Viewing the Clusters Formed
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))


# In[63]:


# Cluster IDs
IDs=[]
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    IDs.append(exemplar)


# In[64]:


IDs


# In[65]:


new_word='DCB_Bank_Officers'
similarity = np.array([distance.levenshtein(w,new_word) for w in IDs])
similarity


# In[66]:


new_word='State_Bank_of_India_Executives'
similarity = np.array([distance.levenshtein(w,new_word) for w in IDs])
similarity


# In[67]:


new_word='Axis_Bank_Board'
similarity = np.array([distance.levenshtein(w,new_word) for w in IDs])
similarity


# In[68]:


# The Plan for new words is use to either use the above similarity measure to keep a certain threshold pertaining to which 
# we either join it in a cluster or form a new cluster for it, but I think when we encounter a new word, it would be better 
# if we can run the model again after adding the new data point because the algorithm would do a better job at finding the 
# appropriate cluster or forming a new cluster of it's own for the new data point.
# Final Note: If I have the data on which I have to work, I can make better models


# In[69]:


# This part of the code deals with the formation of Foriegn Key and Primary Key Relationships within the clusters formed


# In[70]:


# Continuing with our Test Case of Banks we expect our Clustering algorithm to get us the Clusters of Tables of a certain Bank
# Now We are going to find the Primary Key and Foreign Key Relationships between those tables


# In[71]:


# Each List Variable Corresponds to a Table in a Cluster
# The List contains 2 lists in it, the first list corresponds to the Primary Keys of the Table and the second list to the other columns in the table
Branch=[['Branch_ID','Bank_ID'],['Branch_Name','Branch_Address','Bank_Name']]
Customer=[['Customer_ID'],['Full_Name','Age','Sex','Date_of_Joining','Date_of_Leaving','Branch_ID','Bank_ID']]
Account=[['Account_ID'],['Customer_ID','Branch_ID','Balance','Date_of_Opening','Date_of_Closing']]
Loan=[['Loan_ID'],['Account_ID','Purpose','Amount','Branch_ID','Date_of_Loan_Approval']]
Employee=[['Employee_ID'],['Employee_Name','Date_of_Joining','Date_of_Leaving','Employee_Rating','Bank_ID','Branch_ID']]
# The Idea for testing purposes is to take the Primary Keys of the First table and to find which of these primary keys exist
# in the other tables so as to assign them as Foreign Keys in those tables
Cluster=[Branch,Customer,Account,Loan,Employee]
Foriegn_Key_Canditates=Cluster[0][0]
print(Foriegn_Key_Canditates)


# In[72]:


# Now to use the Foriegn_Key_Canditates to make Foreign Keys in other tables 
# Foriegn Keys will be specified seperately as the third element of the list which corresponds to a Table
for i in range(1,len(Cluster)):
    temp=[]
    for j in Cluster[i][1]:
        for k in Foriegn_Key_Canditates:
            if(k==j):
                temp.append(k)
    Cluster[i].append(temp)
for i in Cluster:
    print(i)


# In[73]:


# Printing the Results Obtained:
print('Foriegn Key Canditates obtained from Table 1 in the Cluster are: ')
for i in Foriegn_Key_Canditates:
    print(i,end=' ')
print()
for i in range(1,len(Cluster)):
        print('Foriegn Keys obtained from Table '+str(i)+' in the Cluster are: ')
        for j in Cluster[i][2]:
            print(j,end=' ')
        print()



# In[74]:


# Using the Dictionary Format to store the Clusters as well as the links


# In[75]:


Database={'Cluster1':{'Table1':Branch,'Table2':Customer,'Table3':Account,'Table4':Loan,'Table5':Employee,'Links':{'From':[],'To':[]}}}


# In[76]:


Database


# In[77]:


# Since we take that all links in the cluster start from the first table in the cluster
Database['Cluster1']['Links']['From'].append('Table1')


# In[78]:


# Creating the 'To' List of the Links sub-Dictionary
ans=[]
for i in range(1,len(Cluster)):
    for j in Cluster[i][1]:
        for k in Foriegn_Key_Canditates:
            if(k==j):
                if 'Table'+str(i+1) not in ans:
                    ans.append('Table'+str(i+1))
print(ans)


# In[79]:


Database['Cluster1']['Links']['To']=ans


# In[80]:


Database


# In[81]:


# The Dictionary 'Database' contains the Cluster Organised by a format requested by you.
# You can print and see the 'Database' for yourself.

