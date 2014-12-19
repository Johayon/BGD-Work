import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;




import com.jcabi.ssh.SSH;
import com.jcabi.ssh.Shell;


public class Hostssh2 {

	private String l_host;
	private String dsakey;
	private String output;
	
	public Hostssh2 (String hostname) throws IOException{
		l_host = hostname;
		
		/* Lecture du fichier dsa */
		String dsaPrivatekeyFile = new String("/cal/homes/johayon/.ssh/id_dsa");
		File file = new File(dsaPrivatekeyFile);
	    FileInputStream fis = new FileInputStream(file);
	    byte[] data = new byte[(int)file.length()];
	    fis.read(data);
	    fis.close();
	    //
	    dsakey = new String(data, "UTF-8");
	    output = "";
		
	    /* Connection sshjcabi */
	}
	
	public void exec(String cmd) throws IOException{
			
		Shell shell = new SSH(l_host, 22, "johayon", dsakey);
		String stdout = new Shell.Plain(shell).exec(cmd);
		output =  "Executed " + cmd + " on "+ l_host;
		if (stdout.isEmpty()){
		
		}
		else {
		System.out.println(stdout);
		}
	}
	
	
	public String out(){
		return output;
	}
	
	
		
		
		
}
	
	

