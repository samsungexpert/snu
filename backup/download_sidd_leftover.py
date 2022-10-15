
import os
import requests



def main():
    print("start downloading.....")
    

    addresses = []
    with open('SIDD_URLs_Mirror_2.txt') as file:
        addresses = file.readlines()


    downloaded = []
    with open('downloaded_files.txt') as already:
        downloaded = already.readlines()



    print(len(addresses), len(downloaded))


    aset = set(addresses)
    dset = set(downloaded)
    leftover = aset - dset

    print(len(leftover))
    

    for idx, url in enumerate(leftover):
        print(f'Download {idx+1}/{len(leftover)}')
        print(url)
   
        fname = url.split('/')[-1]

        r = requests.get(url)

        savepath = os.path.join('downloads', fname+'.zip')
        with open(savepath, 'wb') as filew:
            filew.write(r.content)
        
        with open('downloaded_files.txt','a') as filel:
            filel.write(url)

        print(f'Download {idx+1}/{len(addresses)} ---> done')

    print('All done... good job')

if __name__ == '__main__':
    main()
