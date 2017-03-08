/*************************************************************************
 *  Compilation:  javac Students.java
 *  Execution:    java Students < students.txt
 *  Dependencies: StdIn.java
 *  Sample data:  http://introcs.cs.princeton.edu/15inout/students.txt
 *
 *  Reads in the integer N from standard input, then a list
 *  of N student records, where each record consists of four
 *  fields, separated by whitespace:
 *      - first name
 *      - last name
 *      - email address
 *      - which section they're in
 *
 *  Then, print out a list of email address of students in section 4 and 5.
 *
 *  % 130
 *  Sarah Wang twang 7
 *  Austin Taylor actaylor 1
 *  David Rosner drosner 4
 *  Rebecca Allen rebeccaa 7
 *  Rajiv Ayyangar ayyangar 7
 *  Daniel Barrett drbarret 8
 *  Nic Byrd nbyrd 7
 *  Emily Capra ecapra 8
 *  Johnny Clore jclore 7
 *  ...
 *
 * % Section 4
 *  ---------
 *  drosner
 *  jcharles
 *  jph
 *  mlampert
 *  ...
 *
 *  Section 5
 *  ---------
 *  giwank
 *  agrozdan
 *  ajh
 *  akornell
 *  ...
 *
 *************************************************************************/


public class Students {
    public static void main(String[] args) { 

        // read the number of students
        int N = StdIn.readInt();

        // initialize four parallel arrays
        String[] first = new String[N];
        String[] last  = new String[N];
        String[] email = new String[N];
        int[] section  = new int[N];

        // read in the data
        for (int i = 0; i < N; i++) {
            first[i]   = StdIn.readString();
            last[i]    = StdIn.readString();
            email[i]   = StdIn.readString();
            section[i] = StdIn.readInt();
        }

        // print email addresses of all students in section 4
        StdOut.println("Section 4");
        StdOut.println("---------");
        for (int i = 0; i < N; i++) {
            if (section[i] == 4) {
                StdOut.println(email[i]);
            }
        }
        StdOut.println();

        // print email addresses of all students in section 5
        StdOut.println("Section 5");
        StdOut.println("---------");
        for (int i = 0; i < N; i++) {
            if (section[i] == 5) {
                StdOut.println(email[i]);
            }
        }

    }

}
