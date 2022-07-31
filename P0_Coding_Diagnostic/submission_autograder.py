#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('QlpoOTFBWSZTWTAgfhoAPC9fgHkQfv///3////7////7YB1cNvuaHDXTGuOABzrDZrEwoPm7wZegAAaA7FBWEpQOQ6AdA0MCkaGhQ2A1KHQJrOhQQlCEaEZpNMgpk2lPyJ6jJ6SMQ2U8UNDQZBoeptNTQDTIjQQgRqNqZTU9T9I0aNMiPJDQ2oA0GgAGgHA0aMQaNMmEGIDEYmjRo0AaaaAAAAJNFJEIamap6SHqPFNNqHqaPRDymCAANGgADQ0DKomNTAAAmAaAAAJppgAAAAACRIIAQATRMKntT0BRjRT9Tyap6nqeSekaep6hoANNqdpD4Ynyw9oyf52isQZ+Jhf5Wv/SfiTM40FYpOmkZ72sf+2VYqMQSI/22oojBYvyJUOkO/5aVldwLxCwSLARWCQYsnkhKyfzJxyHnthO1sqH1lyw/yaYQgkipYk+RFy2z6cATUPbzrObMP49ze61nO7lVy9qtfyTdn8R6fr09q9mKaVdc/W/GaD33mr1tq0ycKnfH/fqhFWFGkSfL359V/Vtb3u97MO5Cz7vKQhepIwgEqoioqxVgoxAVEQVjFGRYqqQRFnxften0z6Z/F8HmM8vnP+aaW9fMX6mQ88SihJLnD8+I1PsX047HP6axq+1s5okzCxW1pSUoWIPt0OcCu3NW8U9txui7V6ecKctGxstZMZjzc5KV5Skd7+Y4XqVluyYwapWllFWa1Q4hCqqqqq8k+ToPDh267BFkWL2DscDtZUkSEAWZOxZOPZXHm6DYateuQ69/Tg4f4ulKn6rR3gb+d9yHKpVxry7U+eW/IvuB+uEHbnpimhfOs0bcCCSCQASE8Dfue24Czd1wLbZxM1Y5OxgA43RJwNFD8K7nWV1YcgsSCZ+NDQbTnl5Mwd614TfoIWykfLYMqquJWFUVFLJrg5DjowGzQSt3ajmwMa1V8HXR+Af4tYjqqFMdnYhNSwQoPeHKeqn7cDXHttikikZBpfWZZXiijG2DaBsYyXMdtmF+WF9OkRKDERwoy1Sqmv3UWRsc1NOFfydeRNjuNy8rA4Nrm+7ETNn6lThjgZLUx2G28pzf2N2a+gDA4100+gcfzDzDx1TCtuHIyfTnc428qsdGnbMuxrHQjskJRRfkHlvwULl9B20DPe5J44XfPzPuvMsAVEij5M8Kd/08fio3uLwUYiqqlcqZtM78gVMYY6HIicmFt9ju9fv4ZId1BjXWq7eZyyYeqqxpju4no/cb9fR1s+PNfLdsMus8n0Tj1cGXNcM0zrrYuvfcd6bHx2XG5tDiHAqTMdUHkJmZoxjmJjKYjwcbEn0g8L6Ys9lqkl6sXjVkv2PP2/2fb+9vm/9+n3AeHu9fwez21SI/AnPyjzSLqlOLz0AxnIdMWjeu24gcR6tpdvhMZA6sPj+D4Z60vy60M7t77lvESxCJJAYpMEXJxJg5mO9Wn3rVF1w9kwYYoT6HhnV18fuCl4AFTh0qyqWf428ZNNmgtgiQtDCd0UCk61h+7TWBNlcTI0hmrRgCwIqFD9S64zc9Q1FppWK02TKmZ39Pk8NdK6aCHLXKQvAgUyN0cVAl3rQELXBl7y1BgXQ8GLC1h4K1EDOSTbS8ZIAcEDmt2i1Nzy7hpnZtAFobr3S2IOKh9hTuUbIEaALFcvbIW1Qo2mGsE2W5zprfluHXsOuy+Txa7vD6qfpm7P7eKzVb5TH8pmuOFZwL+P8bmlOKjKahbuhQWugQ4WUmtrffaood5DSzOVK+kj4MK1y9mKA7dt7yfnTaE6R3vE7PvIVd+Z0rsSf1QPlLgdqo929STHQsCQT0PUmmI8UN3605Lh8Oe30x4dKa9a5QINRp07IFVqn8OTaTYZ0nSOesjG/wGvTLGuoay/dpod2yhlFJagvyWWyMvjULCq/E2FaHLtg1Xxi0ega7Zxh4RwlpSfgUi/lE6QA2rHHTVKw5bpVszlYpw+kc/UpGYmvHSf4rroPUFjs6A1uHLUqJxL4UWw4nEaRWW2sNdzRyYUcnT5//Lft37DC3tYlrYQx5kPdR2csjFUF08lY4eJV3pEcSyOHM9EUG8sXFCaHUfTaxa/c+dwyH8pzXHn9cjaRsNWIgCArvoUW7yJkElkK4Ljv91CkdjPvw3U9O01yo1mKHagQxQKVoJrlRXUUCeELXpfERYaDhUcOevdm2oKnbSkah7Zr5cqHIyJrKuWqN3Cw4LenhvY6ocPigtOvfF8oFpDyY56mgqXNF/X0Dxa80QEDPMVW7C5BlwZsY4rYdan98SmBcgNX7bwI797syB3NyrkgqBOmOCy8dJYUgcwAq+hhopVd/khh360fcQMkEMWV89LQMPssjQagt0YVsXRNZbLYqpNHwHN4ubKmpcBu1FQCtKa5bFzAguDxn1W2OgMgSy4YVllbTTylirJ7PJldPyQ6pC86IW5+g7/mUTOCToCnDkatfCAvJsOsDYno59rvj3KwGZZiT4PUsIAweYFYUenLOIMn7LiCHuGYftpd9i+QHIvxdtchtzXX9ZelKg0239A0eA7NWyvq6p/V0Qc+H+GO8HvFlueizPhhE554pXdDyx5L9KU7fTublfpscWLnwJ8Hr5jdcZFRh8IkEEcRXt3q5vYQJEkWI25Yrg6aai+59pmidzPZ14zNPON19+RNsxab7SOJYDEQLPoWAWOAdb5NoUfL5x2kn4D8JYI8uNXHyUekEheaF81KQR6TqoAeZK4upLE/dCS8WgVkDPSrIf1ej6/8/9cr70G6QiDRoR8zBtrI29z1+m/zdlTr9vMSWa9vV4H15Glsu+KTkHFz00vUU4wQRPrGs78fRt5eWr4Ffl43wqJqyWfT6qdDWSjgWIG5zDwQye9MVrLuiIW8WdM4QPERN+GKpUpcPSCibQUO32Ori4L6dYs7k3pAfVJtiK5TR7TCOwKBMRtZUwagiqt609USaJPRGPseLSlaDGOdtIpqvaSKOJQACSCOwVvILr/TXY6K4VF9oAEkEL/rrwooCtaSvkEoiAAkghRnABJBF5MIiQDNyo9UOJAJIJz8fYe68H/EBJCPRLw45EvcAkhG7pJdQCSESmP7uyTETyXgIiCHtAREEPH77xgyM285EYYolq5rbJYv9MAPuCREDoZRARhRJRBEkA4STg0MDIjJDoClJIiQKUsIiBeuipZgnBkESAcKWQRDYXGgCJOjPBJG0pg4IIkJOSlkiIXDjSREk/Q+0AkhGWR3d3DzfUAkhH+9T8LdficCnqs9Ph+YCSEXY2z7Le8BJCPh6Mh7wEkIzwk4u1B+TCX72EDiynL+PhXXPKjZmP36daiKlStVJYsQTzKPo9VLqLNekxdIkl0mmqmpFSRVfO+Dn8YdXCd8h3R5YiCnYu5eAv+zXPUuRcTpPJ6EqU3V5To7c7TiUvbHZW6Xg1ll7RNEyY2KbVJtDSZQ8ksYt1Dcdio16puHLmqrFWpTau0rebO5PF4zlaTg4pPB49/ChxBcw7AnVLUlBgiVVVRGylVVtFqskPGuXtUe708Ggvq5ea8ylvL4CQjrxlRQeHbyDcFenxbCt8Xk89uNVrXg5WwUj4bBwuby+Fbc56ivFe7h86/F+QCSEN9/0Pzfo7/Z6/U+vf4AwvG0p8/z4NnW7GZNNdc45G8463NNpbDdWmG8l2jciy2dZ3PZTi9EODBGFwYGggxOoSmKBioDSOmAeCW0SsKU4Tw9fW6KFSMZEXoa2SyxRGJ59iH2O3Dw7a20HS22lRrCBRassIDRNFpqgggg6ta1ECxUd3wJzi2XWNYWMUGLKlSlCwppLrbKMFF0KJRRgyCIUcrr+l+D+oBJCGtz9XcAkhHtASRDc+q/eAkhD2+4+igh6Ci6zKUwmQ2bKmEbqaYZmtaNy/QQJeER422MrZghsjYehGERk4BoQGwgYQEZDgQQokEEZzo4VSsKJeBKIcBiiIGDCRWhDCREHcXE0Qtlh1OCdI1toHASywYIMnROULazA7TEWJin1/yASQQuKdXAhwKqrUqDdanU/SxuXJAumJK1qlsGF5j/ZDuFqbGwX23XYY+M4rEH7/KcmtgwpdCTcapSo7DJfd9odmnvW0tsoHvQ6r8foR8rg1c39oDG63ZE/g/x++4wJm8gt8SC6hdiS/C5LSqMzCUrroNm7/QcQLjflQz5XAe3BO/ff97/snpMJIeoMie0NLIapeo0op4UDhjZNoKDYxQRisMhaI4JsJIwTn/IBJCH9EmuC6KUWL31pMlnqAxRWk5Vy/l/cEjWT4SAJWW7SCQgIgh4aW9rFM1DpOaJwqx81QYXosPG3A5qOwjDO/1gQrl1BOHYskDZ+5l/aD1MOEgC8pigZXNMbrZIxw+LXE3xGZdefNkbA0NJiKZiO3d5pyKALfQlFauLMFLHgzwGSlKUq4Wz4QJTauASQjfae/A+YkZhhHPUrX9Ykvpn9HtJPajTDZkqXCW60U+O84ivHwLC9+/bI8vbHG6vG63mvVKPQdCDEMYIiwrGoUr0FpQ0jFiqahRgkXho8KKNQKTQNMMxopr0eJ62WnEiawnrO1jUxKGoDZYvQjtcVKq4NUxa9UqhPkgMeLr6Enfsh7UoNwNuQ2oHEREBCUtLtDQhstBIiSNNMQ9k9U0mKQmEDYBEbJhy3HV8cnw2SItZDqq9NhRgPvsCN6w+6W6xo5YzmqqOoNiex7XITalCiJSiQybWtGUixuMKbGusrWVSgiqiZxCzis0Gz9AHHBzg0Zm3MiF0iSL7DSXCO6SKvtd3gmRE4WLgn3pxH8xfCqCjpVSQrphJoaY5D9QCSEWoivj39a8y48Vr+/3+CJ0OKAwSP1JNzHHkCJjEwYlLMDqNiTqkHjurOd+Yw/INdfLPG1g2Ew66oWiK3nAtDpQ1O1eVkwDzM8+wM0GeVjs7KbTI7U9vdlThWo6Jv0yi16VKegHxygL453tQB3V26oorxeMLV6xqnTebQjDkJpZm88vQpIBruRD0lNrRvEBVXgXFhEGRau7HfW4KPGUK0aRM5sGqFFO924xBOMK2yq88jK0myCpCVi1g1nMU0pC+fKeyyUg2gR+hK21Hm1ze3xpKrteUlAUp16gMUHUg7/RADQZYiSOq4NndxOYT5VEW9SzLL/1k2wgDzZ9nbvASQjo9CWIMHstTB2g0lxbS5wpdDROhUa5FxGEkDD4QjfxHhWQ5KJNSiQSJkQRkjVephYKOGV9ot4DVC/c+3GBdhSTkIaIihLXQe8pYZsPu60Fb9d1/1EifgIwOTfiazSQU1TLNc9cVU2ijxDfHxMWUC0ViAuttixJkrpiLpAF5YB/CnhYAFtqgv4Cz4jGU7E0DoR1AeYTYJoYhMY0dW66SUvvnB7O3wVntlEvuirQ+wBJCPwy8CieXMsDyFmEwn3nbaZda2yXjUhQYlUtc1Pj7QJTn7CiJp3WkEl6RiGmDY8l36DogxKrswgkqqz5H01rAyBUe9D3dWoCxZPrIAg+FX4idJYJdTLhTLh94t0WayyQFwWAJIRS1KrRuGevXU2Hi0mayEiBpkXaTCB+tiIChEEZ7Hy7CYVVjIuTBEEPz8b955gEkI2pFbDVvRLABJCIPKS8ga/d3ogLd8d65Aizx8qfDkkaCL8uIYccw3gd4xQ39EJdLlqgoZ3eym1OwDBKvcmz9R7rluaBsQTTQHraVQQaAFgiJmBN8zpGmL1kh72qQ1/gcQXkgXUbUvxXYCqibOMuTkRJ5DioxpkpQ5htv37vIDTPyvWRci2NE0NptiBjY00P0oKH+MGliM0g6PtOPm4ZNGOryfgrbjAW9HuvkSBgPKA1lu3wS5zzwLp2m/U5bMa2hI7jIFzASQjTR+n1Mfns1xJHxpm/hvONqcoFvOFw83NzG4Tty8aIsbS0rVtOk2KwrPCzw6lOwkDtGQO141FNWr8uzwrer14I9PWrQcLayUICEU4DkeaUyC5sYIUJUYYshgRlQFCycozkolpRNJCmlmkdkN4glynR1thbcywKjluFIIQRAQxtSgGbcBWCLQ85g8suc+qM17vPzjoCedaUbE2rbJRpTK0FjDMUWFIJaJhKT1et6h7PtM+sMrYA04jwghkH4gJIRKb2ozsCA4Fo2G1iC9pF4X+uMCarthTaFdsMZiJvqtHbkprsOu7r9WVTnz0FuWjA2wQLdKN5ELZd0taPPhptIMs0Y0iV5UTEjGpWlblA3SYC1BgvgHfJ6XELHjrFZJGlERKetfNyyO5mrbdB2d0kjdRZ6RlcAkhFNqk5xBOFDaQNJgMjixpsNb2Yb628DAw3YkupfsioZo5gYgMV2l6DjaP57Qm8QD6RkrJg2afTlufIwXVt9lxgnyCGDMIWQ2NjbG2MUWIiBqNLeDitL7M7hyp+GBWM4WWBRgT2ZslJCOb/nWCbadxewJSEwuMNOu23plMc3HE3XooA2i5F8nIgaDAgRJAQKBNZzJeQybSSKIDrskyzJEvQKziBqnZw4PeOoMDYNa1367sExGpkhsBsAwYpsEKy+1JPICSEOpLHsfZn7j3fR31X38+2AkhHJLh3/1kj2zCFsFQbbYS8xvOfA42/xbYtAVp5Aj7AUHhZdF9m1n6Us9ekcVwhS1vU0jnSkamgbEiUoSJhys4nJfhZYrESZoF8eH69vHlS/20aBQXF22aqFqHb6vEVA0aRYCLkYcMw82dbUk0KP5UyFsiL/1Fe8DBNu3xxxz4gwE56BU5vPubjrBf2Pkx9R0DifpM3hWvVKjPqYzY2KlMnZeXWZbvn5zjyqlCwxK5DpeW2jw0rDYtDunxvR3+5dO/lb414qZ8Rjx5k8mycVqhRVtTlKqdS1daIo0KjlaGRmYpQ/V51yyjKhQayLHkkB2gxxNhsByPXee/6Pmn8j7/ik8J5Hq8lTVhS6ibQRbBw0hk1YKKQF+PIaH8vDs4bB/s2mCMIgZEYyiHA4gIbh1DFo0QGQSrQZWjqSPpD5vm/OPFFq3SEieGL5Wza7Wxhqdl1nUIuRiBRfDpjdYZIA4vjPKpB/UAkhEqCMCbLbHjAIkx5p22r5pR8mh4t7HDBZEh7e/juyHSVs8QPpAMWYYoC/uAJ3XLB5DJJ2pc8qypBSLogvRjyFmKnVifOWomdoey9ZjHTKEku538M66jmAkhDMfGtJzKplig9Zsqigh4pgDADZai0Rd4+tUwexYcbccA/JQGsJAQFrzAM2mxz8X4LCBQk1DCrBeT3UtDHzic2aHAcgfcmmKVdG2FCZ52bLK3j9NayllBY5qjIjzxLe4Tw0NVoF0FIJXqL7LcneMtZZdIJANFyCH4SMxEUk4gKOxkfWAkhGOWCxsCQtiVuvvfU56g4txaMYyQQqrvZ6srvmHdfW+xEwxvn4ICQGi4BUnQ+F1/jPG9ItwhMaAbWHDTfosdptjeOdprFnYDYmCJKEAweQ4yiN4EAay67nKt5ri/cBpgZ4zbrYiKY11PMAOEfwlXu11GBc9jB95bQwAhz38Fv4DjhggCVBU9KuE6nfXUogdZuZLXS3vtxDx9VHo7MTo6cB03hNBaA7t2iGoCBQsWigejMRcj0hBuj2wALSB6DTftrGJuHEglSU5ZDhlHXKEZAJIQ+u813Yo+xhD06MH2ZZhS7AEWpMDBjUkBya7CyuAaK8HQyTGy+jTvl+AXqiioD0DE8ty3Gh71gScOatnJKInEsUqAncyYi5ImVjmdnJE0sHah7sjNtmTEQge2JD5pGqxOQATuW8sVeSBgXAJIRdUOqYBTDUuVpKFM+QiMoPzH6QdZZLph0Bi8ng7Dqqhr87Th3KJylJFtq107LJAB2BYBDQLDgbKLLXljnFKaCbMhVoMbdKqoESlLMIy4OC3oASQhznph2vGOxN77fB0xKOWc8POTliwmvLGlDgVFz+wBJCKFtyhKgsbQlqOh9uzKwXsaTD0AJIR2d54Rv6SUk5QraUpMStYiQQqSmiSBohnY1zcpHkAkhHorbaKprR3ckTMPnrDNfyMhqEby9gPs+ICSEZmhjYUXyZzC1/wce5+dfyGVOv0ccafLNrPWnr9y+xey/uyTpCd44cTjpAf+LuSKcKEgYED8NA')))

