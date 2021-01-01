import pathlib
import csv
import subprocess
import time

"""
where balanced_train_segments.csv, unbalanced_train_segments.csv,
and eval_segments.csv are
"""

metadata_dir = pathlib.Path('../../audioset')

# where to save the downloaded audio files

dst_dir = pathlib.Path('../../data/raw/0_AUDIOSET')

"""
class labels can be found in the class_labels_indices.csv file, which
can be found here:

http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

the desired_labels list is used to choose the AudioSet files to download
"""

desired_labels = ['/m/09x0r', # speech
                  '/m/015lz1' # singing
                 ] 

# to track the number of successfully downloaded files

num_downloaded = 0

"""
iterate over each of balanced_train_segments.csv,
unbalanced_train_segments.csv, and eval_segments.csv
"""

start_epoch = time.time()

for csv_file in metadata_dir.iterdir():
    print('\nReading {}...'.format(csv_file.name))
    # start reading csv file
    fp = open(csv_file)
    csv_reader = csv.reader(fp,
                            delimiter = ',',
                            skipinitialspace = True)
    # skip first three rows since they do not include metadata
    for _ in range(3):
        next(csv_reader)
    # start iterating from 4th row
    for i,row in enumerate(csv_reader):
        # YouTube video ID of audio snippet
        yt_id = row[0]
        # start time of audio snippet in seconds
        start = row[1]
        # end time in of audio snippet seconds
        end = row[2]
        # labels associated with audio snippet
        snippet_labels = row[3].split(',')
        # check if any of the desired labels are associated with this
        # audio snippet. The download flag indicates whether to download
        # this audio snippet or not
        download = False
        for label in desired_labels:
            # if a desired label is indeed associated with this audio
            # snippet, there is no need to check the other desired labels
            if label in snippet_labels:
                download = True
                break
        """
        download the YouTube audio snippet using youtube-dl, which can
        be downloaded from here:
            
        https://youtube-dl.org/
        
        And ffmpeg, which can be downloaded from here:
        
        https://ffmpeg.org/
        """
        
        if download:
            """
            YouTube uses the Opus audio codec. The following cmd.exe
            (Windows) command returns all audio and video URLs associated
            with a YouTube video. This command includes the option
            -f bestaudio[ext-webm] to choose the opus audio URL with the
            highest bitrate. A list of the available YouTube itag codes can
            be found here:
                
            https://gist.github.com/sidneys/7095afe4da4ae58694d128b1034e01e2
            
            NOTE: the double quotation marks that enclose the YouTube URL
            are important. However, if a sequence is passed to the
            subprocess.run() function, then quotation marks are not
            needed. Moreover, if this same command is to be executed
            directly in cmd.exe, then quotation marks surroudning the
            YouTube URL are needed.
            """
            ytdl_cmd = ['youtube-dl',
                        '-g',
                        '-f',
                        'bestaudio[ext=m4a]',
                        'https://www.youtube.com/watch?v='+yt_id]
            
            # if the YouTube video is no longer available or has been taken
            # down by its author, skip to the next iteration of the for
            # loop
    
            try:
                """
                execute in cmd.exe and return result as a byte-string. The resulting
                string can be searched for its itag code using ctrl+f. This is done by
                searching: "itag=ITAG_CODE" (without quotation marks). For example,
                searching itag=251 will check if the resulting audio url is for the
                opus audio file with a sampling rate of 48 kHz and a bitrate of
                160 kbps.
                
                The following cmd.exe command:
                    
                youtube-dl -F "YOUTUBE_URL"
                
                lists all the audio and video formats available for download with their
                associated itag codes. For example:
                    
                youtube-dl -F "https://www.youtube.com/watch?v=x_R-qzjZrKQ"
                """
                
                ytdl_out = subprocess.run(args = ytdl_cmd,
                                          capture_output = True,
                                          encoding = 'utf-8',
                                          check = True)
            
            except subprocess.CalledProcessError:
                # audio URL cannot be obtained
                continue
        
            # remove the last newline character to get the audio URL
        
            audio_url = ytdl_out.stdout[:-1]
            
            # convert start time to 'HH:MM:SS' format for the ffmpeg
            # command

            start_time = time.strftime("%H:%M:%S",
                                       time.gmtime(int(float(start))))
            
            # compute the snippet length in seconds and convert it to
            # 'HH:MM:SS' format for the ffmpeg command
            
            snippet_length = time.strftime("%H:%M:%S",
                                           time.gmtime(int(float(end)-
                                                           float(start))))
            
            """
            create the appropriate ffmpeg command and execute it in
            cmd.exe.
            
            NOTE: double quotation marks are required to enclose the
            audio_url. See:
                
            https://unix.stackexchange.com/questions/427891/ffmpeg-youtube-dl
            
            for details. This command won't work otherwise. However, if
            the ffmpeg command is passed as a sequence to the
            subprocess.run() function, then it is not required to enclose
            audio_url with quotation marks.
            
            If the ffmpeg command is to be executed directly in cmd.exe,
            then the quotation marks are required.
            """
            
            dst = dst_dir / (yt_id + '_' + str(int(float(start))) + '_'
                             + str(int(float(end))) + '.m4a')
            
            ffmpeg_cmd1 = ['ffmpeg',
                           '-ss',
                           start_time,
                           '-i',
                           audio_url,
                           '-t',
                           snippet_length,
                           '-c',
                           'copy',
                           str(dst)]
        
            try:
                
                # execute command in cmd.exe using ffmpeg.exe
                
                ffmpeg_out1 = subprocess.run(args = ffmpeg_cmd1,
                                             capture_output = True,
                                             encoding = 'utf-8',
                                             check = True)
            
            except subprocess.CalledProcessError:
                # audio cannot be downloaded
                continue
        
            # convert downloaded .m4a file to a .wav file
            
            ffmpeg_cmd2 = ['ffmpeg',
                           '-i',
                           str(dst),
                           str(dst.with_suffix('.wav'))
                           ]
            
            # execute command in cmd.exe using ffmpeg.exe
                
            ffmpeg_out2 = subprocess.run(args = ffmpeg_cmd2,
                                         capture_output = True,
                                         encoding = 'utf-8',
                                         check = True)
            
            # delete the .m4a file as it is no longer needed
            
            dst.unlink()
            
            # track progress
            
            num_downloaded += 1                
            print('\rRow {}, Downloaded {} files'.format((i+1),
                                                         num_downloaded),
                  end='',flush=True)
    
    fp.close()

end_epoch = time.time()

print('\nTime elapsed: {}'.format(time.strftime(
                                  "%H:%M:%S",time.gmtime(end_epoch-
                                                         start_epoch))))