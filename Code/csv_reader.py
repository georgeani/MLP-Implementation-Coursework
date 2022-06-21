import csv


def standardisedTraining():
    output = list()
    with open("../Standardised Training.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            output.append({
                'Area': float(row['ï»¿Area Standardised']),
                'BIHOST': float(row['BIHOST Standardised']),
                'FARL': float(row['FARL Standardised']),
                'FPEXT': float(row['FPEXT Standardised']),
                'LDP': float(row['LDP Standardised']),
                'PROPWET': float(row['PROPWET Standardised']),
                'RMED': float(row['RMED-1D Standardised']),
                'SAAR': float(row['SAAR Standardised']),
                'Index': float(row['Index Flood Standardized'])
            })
    csv_file.close()
    return output


def standardisedValidation():
    output = list()
    with open("../Validation Data.csv", mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for row in reader:
            output.append({
                'Area': float(row['ï»¿Area Standardised']),
                'BIHOST': float(row['BFIHOST Standardized']),
                'FARL': float(row['FARL Standardised']),
                'FPEXT': float(row['FPEXT Standardised']),
                'LDP': float(row['LDP Standardised']),
                'PROPWET': float(row['PROPWET Standardised']),
                'RMED': float(row['PMED-1D Standardised']),
                'SAAR': float(row['SAAR Standardised']),
                'Index': float(row['Index Flood Standardised'])
            })
    csv_file.close()
    return output


def standardised_testing():
    output = list()
    with open("../Testing Data.csv", mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for row in reader:
            output.append({
                'Area': float(row['ï»¿Area Standardised']),
                'BIHOST': float(row['BFIHOST Standardised']),
                'FARL': float(row['FARL Standardised']),
                'FPEXT': float(row['FPEXT Standardised']),
                'LDP': float(row['LDP Standardised']),
                'PROPWET': float(row['PROPWET Standardised']),
                'RMED': float(row['RMED-1D Standardised']),
                'SAAR': float(row['SAAR Standardised']),
                'Index': float(row['Index Flood Standardised'])
            })
    csv_file.close()
    return output


def input_limited_training():
    output = list()
    with open("../Standardised Training.csv", mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for row in reader:
            output.append({
                'Area': float(row['ï»¿Area Standardised']),
                'LDP': float(row['LDP Standardised']),
                'SAAR': float(row['SAAR Standardised']),
                'Index': float(row['Index Flood Standardized'])
            })
    csv_file.close()
    return output


def input_limited_validation():
    output = list()
    with open("../Validation Data.csv", mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for row in reader:
            output.append({
                'Area': float(row['ï»¿Area Standardised']),
                'LDP': float(row['LDP Standardised']),
                'SAAR': float(row['SAAR Standardised']),
                'Index': float(row['Index Flood Standardised'])
            })
    csv_file.close()
    return output


def input_limited_testing():
    output = list()
    with open("../Testing Data.csv", mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for row in reader:
            output.append({
                'Area': float(row['ï»¿Area Standardised']),
                'LDP': float(row['LDP Standardised']),
                'SAAR': float(row['SAAR Standardised']),
                'Index': float(row['Index Flood Standardised'])
            })
    csv_file.close()
    return output


if __name__ == '__main__':
    print(input_limited_testing())
