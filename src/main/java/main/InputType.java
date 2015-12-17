package main;

import java.io.Serializable;

public class InputType implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int label;
	private String content;

	public InputType() {
		super();
		// TODO Auto-generated constructor stub
	}

	public InputType(int label, String content) {
		super();
		this.label = label;
		this.content = content;
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public String getContent() {
		return content;
	}

	public void setContent(String content) {
		this.content = content;
	}

}
