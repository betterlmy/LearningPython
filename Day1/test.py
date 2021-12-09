# author:TYUT-Lmy
# date:2021/12/9
# description:
import requests as r


def main():
    my_url = r"https://www.baidu.com"
    my_data = {"name": "lmy"}
    request = r.post(my_url, my_data)



if __name__ == '__main__':
    main()
