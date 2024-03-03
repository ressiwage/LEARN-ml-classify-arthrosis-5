Public Function OpenReport(C_REPORT_EDIT_DOCUMENT_ID: String): IPrxReport;
Var
	report: IPrxReport;
	MB: IMetabase;
	MObj: IMetabaseObjectDescriptor;
	Target: IUiCommandTarget;

Begin
	MB := MetabaseClass.Active; 
	MObj := FindInRepo(C_REPORT_EDIT_DOCUMENT_ID, MB);
	Target := WinApplication.Instance.GetObjectTarget(MObj);
	Target.Execute("Object.Open", Null);
	report := MObj.Bind As IPrxReport;
	Return report;
End Function OpenReport;


Public Sub CreateNewDocument(report: IPrxReport);
Var
	range: ITabRange;
	tabSheet: ITabSheet;
	Controls: IPrxControls;
	tableSelection: IDimSelectionSet;
	i, j, k:Integer;
	attrs: IDimAttributesInstance;
	documentKey: Integer;
	dict: IRdsDictionaryInstance;
	documentRecord: IRdsDictionaryElement;
	newReport: IPrxReport;
Begin
	{//WinApplication.InformationBox(report.Controls.Item(0).Name);
	range := (report.ActiveSheet As IPrxTable).TabSheet.View.Selection.Range;
	tabSheet := (report.ActiveSheet As IPrxTable).TabSheet;
	
	//WinApplication.InformationBox(range.Left.ToString );
	//WinApplication.InformationBox(tabSheet.CellValue(range.Bottom, 0) );
	//WinApplication.InformationBox(tabSheet.Regions.Item(0).Range.Right.ToString );
	//WinApplication.InformationBox(tabSheet.Cell(tabSheet.View.Selection.FocusedRow, C_TABLE_DOCUMENT_IDX).Text);
	
	documentKey := Integer.Parse(tabSheet.Cell(tabSheet.View.Selection.FocusedRow, C_TABLE_DOCUMENT_IDX-1).Value);
	
	dict:=GetDict(C_RDS_DOCUMENTS);
	
	
	WinApplication.InformationBox(GetAttrOfElement("OBJECT_ID", documentRecord, dict));
	
	newReport := OpenReport("REPORT_EDIT_DOCUMENT");
	//newReport.DataArea.Slices.Count.ToString атрибуты
	WinApplication.InformationBox(report.Controls.Count.ToString);
	//newReport.Controls.Count := GetAttrOfElement("OBJECT_ID", documentRecord, dict);}
	
	documentKey := Integer.Parse(tabSheet.Cell(tabSheet.View.Selection.FocusedRow, C_TABLE_DOCUMENT_IDX-1).Value);
	dict:=GetDict(C_RDS_DOCUMENTS);
	documentRecord := FindRecordInDict(dict, "KEY", documentKey);
	
	CreateDocumentRds(dict, GetAttrOfElement("OBJECT_ID", documentRecord, dict));
	report.RefreshDataSources;
	
End Sub CreateNewDocument;


Public Sub EditDocument(report: IPrxReport);

Begin
	WinApplication.InformationBox("заглушка");

End Sub EditDocument;


Public Sub DeleteDocument(report: IPrxReport);

Begin
	WinApplication.InformationBox("заглушка");

End Sub DeleteDocument;


Public Sub SendToApprovalDocument(report: IPrxReport);

Begin
	WinApplication.InformationBox("заглушка");

End Sub SendToApprovalDocument;


Public Sub ApproveDocument(report: IPrxReport);

Begin
	WinApplication.InformationBox("заглушка");

End Sub ApproveDocument;


Public Sub DisapproveDocument(report: IPrxReport);

Begin
	WinApplication.InformationBox("заглушка");

End Sub DisapproveDocument;

Public Sub getDataFromDict(dictId:String);
Var
	meta:IMetabase;
	dict:IRdsDictionaryInstance;
	i:Integer;
	record: IRdsDictionaryElement;
Begin
	meta := MetabaseClass.Active; 
	dict :=  DictionaryExt.Open( FindInRepo(dictId,meta));
	dict.FetchAll := True;
	For i := 1 To dict.Elements.Count-1 Do
		record := dict.Elements.Item(i);
		
		Debug.WriteLine(GetAttrOfElement("KEY", record, dict));
		Debug.WriteLine(GetAttrOfElement("NAME", record, dict)); 
		
	End For;
	
End Sub getDataFromDict;

Public Function FindRecordInDict(dict:IRdsDictionaryInstance; attrName:String; attrValue:Variant):IRdsDictionaryElement;
Var
	meta:IMetabase;
	i:Integer;
	record: IRdsDictionaryElement;
Begin
	meta := MetabaseClass.Active; 
	dict.FetchAll := True;
	For i := 1 To dict.Elements.Count-1 Do
		record := dict.Elements.Item(i);
		
		If GetAttrOfElement(attrName, record, dict) = attrValue Then
			Return record;
		End If;
		
	End For;
	Return Null;
	
End Function FindRecordInDict;

Public Function GetDict(id:String):IRdsDictionaryInstance;
Var 
	meta: IMetabase;
Begin
	meta := MetabaseClass.Active; 
	Return DictionaryExt.Open( FindInRepo(id,meta));
End Function GetDict;


Function GetAttrOfElement(attrName:String; record:IRdsDictionaryElement; dict:IRdsDictionaryInstance):Variant;
Begin
	Return record.Attribute(dict.Attributes.FindById(attrName).Key);
End Function GetAttrOfElement;


Function FindInRepo(id:String; meta:IMetabase):IMetabaseObjectDescriptor;
Var
	FInfo: IMetabaseObjectFindInfo;
	MDescs: IMetabaseObjectDescriptors;
Begin
	FInfo :=  meta.CreateFindInfo;
	FInfo.Text :=  id;
	FInfo.Attribute :=  FindAttribute.Ident;
	FInfo.ScanNestedNamespaces :=  True;
	MDescs :=  meta.Find(FInfo); //если здесь ошибка значит не найлено
	If MDescs.Count = 0 Then
		Return Null;
	End If;
	Return MDescs.Item(0);
End Function FindInRepo;


Public Function CreateDocumentRds(dict: IRdsDictionaryInstance; objId: Integer): Integer;
Var
	attrs: IRdsAttributes;
	DictInst: IRdsDictionaryInstance;
	elements: IRdsDictionaryelements;
	data: IRdsDictionaryElementdata;
	key: Integer;
Begin
	attrs := dict.Dictionary.Attributes;
	elements := dict.elements;
	data := elements.Createdata;
	data.Value(data.AttributeIndex(attrs.Name.key)) := "Новая карточка документа";
	data.Value(data.AttributeIndex(attrs.FindById("OBJECT_ID").Key)) := objId;
	key := elements.Insert(-2, data);
	Return key;
End Function CreateDocumentRds;