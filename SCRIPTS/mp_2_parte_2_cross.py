        archivo = pd.DataFrame()
        archivo = archivo[[atributo,'No Dengue','NG,no sig','NG,sig aler','Dengue  Grave']]
        writer = ExcelWriter()  
        archivo.to_excel(writer, sheet_name = atributo,index=False)
        writer.save()

        data = { atributo: valores,
                    'No Dengue': temp1,
                    'NG,no sig': temp2,
                    'NG,sig aler': temp3,
                    'Dengue  Grave': temp4}