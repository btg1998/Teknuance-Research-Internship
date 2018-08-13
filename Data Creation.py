# This Python Script is to be used for Data Generation of Common Table Names

a=['Customer','Account','Branch','Loan','Employee']
b=['Punjab_National_Bank',
   'State_Bank_of_India',
   'Axis_Bank',
   'Dhanalaxmi_Bank',
   'Union_Bank',
   'HDFC_Bank',
   'ICIC_Bank',
   'Canara_Bank',
   'Bank_of_Baroda',
   'Yes_Bank',
   'Kotak_Mahindra_Bank',
   'IDFC_Bank',
   'DCB_Bank',
   'Bandhan_Bank']
final=[]
for i in a:
    for j in b:
        final.append(j+'_'+i)

# This prints the final list of table names which can be used as a Test Case
print(final)
