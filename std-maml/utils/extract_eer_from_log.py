#!/usr/bin/env python


genres = ['entertainment', 'interview', 'singing', 'movie', 'vlog', 'live_broadcast', 'speech', 'drama']


def main():
    
    #logfile = '/work103/kangjiawen/091620-maml-cn2/results/0921-2/infer-mc-mct-genre.log'
    #logfile = '/work103/kangjiawen/091620-maml-cn2/results/0921-2/infer-pair-mct-genre.log'
    logfile = '/work103/kangjiawen/0923-std-maml/results/0924/infer-mc-maml-genre.log'
    #logfile = '/work103/kangjiawen/0923-std-maml/results/0923/infer-pair-maml-genre.log'
    print(logfile)
    
    entertain = Results('entertainment')
    interview = Results('interview')
    singing = Results('singing')
    movie = Results('movie')
    vlog = Results('vlog')
    live = Results('live_broadcast')
    speech = Results('speech')
    drama = Results('drama')
    
    genre_group = [entertain,interview,singing,movie,vlog,live,speech,drama]
    
    genre_group = run(logfile, genre_group)

    for genre in genre_group:
        print('\n{}'.format(genre.name))
        print('Cosine')
        for con in genre.cosine_eer:
            print(con)
        print('PLDA')
        for con in genre.plda_eer:
            print(con)

    return

            
def run(log_file, group):
    logfile = log_file
    genre_group = group

    with open(logfile, 'r') as f:
        lines = f.readlines()

    for i,con in enumerate(lines):
        for genre in genre_group:
            if genre.name in con:
                eer = lines[i+1].strip().split(' ')[2]
                if 'Cosine' in lines[i+1]:
                    genre.cosine_eer.append(eer)
                elif 'LDA' in lines[i+1]:
                    genre.plda_eer.append(eer)
    return genre_group


class Results(object):
    def __init__(self, name):
        self.name = name
        self.cosine_eer = []
        self.plda_eer = []


if __name__ == '__main__':
    main()


