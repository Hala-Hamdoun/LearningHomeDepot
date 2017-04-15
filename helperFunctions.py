#Helper Functions


def loadData(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [];
    productTriples = []
    for product in lines:
        products.append(product.split(","));
    
    #parses the dates to add zeroes for non-existent dates
    for i in range ( 0, len( products ) ):
        products[i] = dateParser( products[i] )

    for product in products:
        for i in range(len(product)):
            if( product[i] ):
                product[i] = product[i].split("|");
            
    for productNumber in range(len(products)):
        product = products[productNumber];
        for i in range(len(product)-4):
            productTriples.append([product[i],product[i+1],product[i+2],product[i+3],product[i+4]]);
            
    return productTriples;

def printRow(file, row):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [];
   
    for product in lines:
        products.append(product.split(","));
        
    return_row = [];

    for i in range( 0, len( products[row] ) ):
        return_row.append( products[row][i] );

    return return_row

def numDates(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [];

    for product in lines:
        products.append(product.split(","));
        
    return_row = [];

    for i in range( 0, len( products ) ):
        return_row.append( len(products[i]) );

    return return_row

#takes the datestring and converts it to a number representing the day (1-365)
#value on the right side is untouched ex: "2016-01-12|14" becomes "12|14"
def dateParser(productArray):
    monthOffset = [ 0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 ];
    parsed_array = [None] * 365;

    for i in range( 0, 365 ):
        if parsed_array[i] == None:
            parsed_array[i] = str(i + 1) + "|0";

    for i in range( 0, len( productArray ) ):
        split_string = productArray[i].split('|');
        numbers_string = split_string[0].split('-');
        parsed_array[ monthOffset[ int( numbers_string[1] ) ] + int( numbers_string[2] ) - 1 ] = str( monthOffset[ int( numbers_string[1] ) ] + int( numbers_string[2] ) ) + '|' + split_string[1];

    return parsed_array

